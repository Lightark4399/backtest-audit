"""Tests for the alignment audit.

These are the checks that turn "we believe the evaluation is aligned" into
something a build can verify. Two failure modes matter equally:

* **Missed detection** -- a genuinely misaligned panel passing. Tested by
  constructing panels with a deliberate off-by-one.
* **False alarm** -- an honest panel failing. Tested by constructing panels
  correct by construction. This direction matters just as much: a check that
  fires on correct input teaches its users to ignore it, at which point it
  protects nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit.audits.alignment import (
    SHUFFLE_TOLERANCE,
    alignment_summary,
    demeaned_label_autocorrelation,
    label_autocorrelation,
    run_alignment_audit,
    run_shift_test,
    run_shuffle_test,
    shift_labels,
    shuffle_labels_within_date,
)
from audit.metrics.ic import demeaned_ic, rank_ic
from audit.panel import DATE, ENTITY, LABEL, PRED, Panel
from audit.synthetic import generate_panel


def _misaligned_panel(skill: float = 0.6) -> Panel:
    """A panel whose predictions are shifted one date out of place.

    Built by moving each entity's prediction one row earlier, so the prediction
    stored against date t is really the forecast for t+1. This is the concrete
    form of a lag applied in the wrong direction.
    """
    p, _ = generate_panel(skill=skill)
    d = p.data.sort_values([ENTITY, DATE], kind="mergesort").copy()
    d[PRED] = d.groupby(ENTITY, sort=False)[PRED].shift(-1)
    d = d.loc[d[PRED].notna()].reset_index(drop=True)
    return Panel(data=d, train_end=p.train_end, label_name=p.label_name)


# ----------------------------------------------------------------------
# Perturbation mechanics
# ----------------------------------------------------------------------
def test_shuffle_preserves_every_marginal():
    """Only the entity-to-label correspondence may change.

    If the shuffle altered the set of labels on a date, a collapsed IC could be
    explained by the distribution changing rather than by the pairing breaking,
    and the test would prove nothing.
    """
    p, _ = generate_panel(skill=0.5)
    s = shuffle_labels_within_date(p, seed=1)

    assert len(s.data) == len(p.data)
    for date in p.dates[:10]:
        a = np.sort(p.data.loc[p.data[DATE] == date, LABEL].to_numpy())
        b = np.sort(s.data.loc[s.data[DATE] == date, LABEL].to_numpy())
        np.testing.assert_allclose(a, b)
    # Predictions must be untouched
    np.testing.assert_allclose(p.data[PRED].to_numpy(), s.data[PRED].to_numpy())


def test_shuffle_is_deterministic_given_seed():
    """An audit whose verdict changes between runs cannot be checked by a reviewer."""
    p, _ = generate_panel(skill=0.5)
    a = shuffle_labels_within_date(p, seed=7).data[LABEL].to_numpy()
    b = shuffle_labels_within_date(p, seed=7).data[LABEL].to_numpy()
    np.testing.assert_allclose(a, b)

    c = shuffle_labels_within_date(p, seed=8).data[LABEL].to_numpy()
    assert not np.allclose(a, c), "different seeds should give different permutations"


def test_shuffle_actually_permutes():
    p, _ = generate_panel(skill=0.5)
    s = shuffle_labels_within_date(p, seed=3)
    changed = (p.data[LABEL].to_numpy() != s.data[LABEL].to_numpy()).mean()
    assert changed > 0.8, "shuffle left most labels in place"


def test_shift_takes_label_from_the_intended_date():
    dates = pd.bdate_range("2022-01-03", periods=4)
    frame = pd.DataFrame(
        {
            "entity_id": ["A"] * 4 + ["B"] * 4,
            "event_date": list(dates) * 2,
            "prediction": [0.0] * 8,
            "label": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    p = Panel.from_frame(frame, train_end=dates[1])

    fwd = shift_labels(p, offset=1)
    a = fwd.data[fwd.data[ENTITY] == "A"].sort_values(DATE)
    assert list(a[LABEL]) == [2.0, 3.0, 4.0]  # each row now holds the NEXT label

    back = shift_labels(p, offset=-1)
    a2 = back.data[back.data[ENTITY] == "A"].sort_values(DATE)
    assert list(a2[LABEL]) == [1.0, 2.0, 3.0]  # each row holds the PREVIOUS label


def test_shift_does_not_borrow_across_entities():
    """Entity A's last row must not receive entity B's first label."""
    dates = pd.bdate_range("2022-01-03", periods=3)
    frame = pd.DataFrame(
        {
            "entity_id": ["A"] * 3 + ["B"] * 3,
            "event_date": list(dates) * 2,
            "prediction": [0.0] * 6,
            "label": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        }
    )
    p = Panel.from_frame(frame, train_end=dates[0])
    shifted = shift_labels(p, offset=1)
    assert 100.0 not in set(shifted.data.loc[shifted.data[ENTITY] == "A", LABEL])


def test_shift_by_zero_is_rejected():
    p, _ = generate_panel(skill=0.3)
    with pytest.raises(ValueError, match="non-zero"):
        shift_labels(p, offset=0)


# ----------------------------------------------------------------------
# Shuffle test verdicts
# ----------------------------------------------------------------------
def test_shuffle_test_passes_on_honest_panel():
    p, _ = generate_panel(skill=0.6)
    check = run_shuffle_test(p)
    assert check.passed is True
    assert abs(check.perturbed_ic) < SHUFFLE_TOLERANCE


def test_shuffle_collapses_ic_from_a_high_baseline():
    """The collapse must be from a genuinely high starting point to mean anything."""
    p, _ = generate_panel(skill=0.6)
    check = run_shuffle_test(p)
    assert check.baseline_ic > 0.5
    assert check.drop_ratio > 0.9


def test_shuffle_test_detects_a_self_correlating_metric():
    """A panel where prediction IS the label: shuffling must destroy the score.

    This stands in for the class of bug where the evaluation ends up correlating
    a series with itself. Here the pairing is real, so the shuffle should still
    collapse it -- what would fail this test is an implementation that shuffles
    predictions and labels together.
    """
    p, _ = generate_panel(skill=0.6)
    identical = p.replace_prediction(p.data[LABEL])
    check = run_shuffle_test(identical)
    assert check.baseline_ic == pytest.approx(1.0, abs=1e-9)
    assert check.passed is True


# ----------------------------------------------------------------------
# Shift test verdicts: detection and, equally, no false alarms
# ----------------------------------------------------------------------
def test_forward_shift_passes_on_honest_panel():
    """No false alarm on a panel that is correct by construction."""
    p, _ = generate_panel(skill=0.6)
    check = run_shift_test(p, offset=1)
    assert check.passed is True, f"false alarm: {check.verdict}"


def test_backward_shift_is_diagnostic_never_a_verdict():
    """The backward shift must not assert pass or fail, on any input.

    A forecaster of a persistent target is usually built from lagged labels, so
    its prediction resembles the label it was built from more than the one it
    forecasts. Scoring against the previous date therefore routinely beats
    correct alignment -- on the clean example pipeline it raises the demeaned IC
    from 0.71 to 0.97 -- and gating on that would fail nearly every honest
    autoregressive model.
    """
    for skill in (0.0, 0.3, 0.6):
        p, _ = generate_panel(skill=skill)
        check = run_shift_test(p, offset=-1)
        assert check.passed is None
        assert check.detail["diagnostic_only"] is True


def test_diagnostic_checks_do_not_drag_down_the_summary():
    """A clean run must read as all-passed despite the diagnostic being None."""
    p, _ = generate_panel(skill=0.6)
    summary = alignment_summary(run_alignment_audit(p))
    assert summary["all_passed"] is True
    assert summary["n_gated"] < summary["n_checks"]


def test_shift_test_detects_an_off_by_one():
    """A prediction misplaced by one date must be caught."""
    bad = _misaligned_panel()
    check = run_shift_test(bad, offset=1)
    assert check.passed is False
    assert check.perturbed_ic > check.baseline_ic


def test_shift_test_inconclusive_when_there_is_no_signal():
    """With demeaned IC at zero there is nothing whose alignment could be tested.

    Reporting PASS here would be misleading: the model has no date-specific
    content to be aligned, so the check has nothing to say.
    """
    p, _ = generate_panel(skill=0.0)
    check = run_shift_test(p, offset=1)
    assert check.passed is None
    assert "INCONCLUSIVE" in check.verdict


def test_shift_test_runs_on_demeaned_series_by_default():
    """The level is time-invariant, so a raw-IC shift test is nearly blind.

    This asserts the motivation for the default: shifting moves the demeaned IC
    substantially more than it moves the raw IC. If a future change reverted the
    default to raw, this test would record the loss of sensitivity.
    """
    p, _ = generate_panel(skill=0.6)

    raw_base = rank_ic(p).mean
    raw_shift = rank_ic(shift_labels(p, 1)).mean
    raw_drop = (raw_base - raw_shift) / abs(raw_base)

    dm_base = demeaned_ic(p).mean
    dm_shift = demeaned_ic(shift_labels(p, 1)).mean
    dm_drop = (dm_base - dm_shift) / abs(dm_base)

    assert dm_drop > 2 * raw_drop, (
        f"demeaning should sharpen the shift test: raw drop {raw_drop:.3f}, "
        f"demeaned drop {dm_drop:.3f}"
    )
    assert run_shift_test(p, offset=1).detail["basis"] == "demeaned IC"


# ----------------------------------------------------------------------
# Persistence context
# ----------------------------------------------------------------------
def test_demeaned_autocorrelation_is_below_raw():
    """Removing the constant level must reduce measured persistence.

    The raw figure is inflated by the entity level: two observations of the same
    entity look alike largely because of what the entity is. The demeaned figure
    measures the part a shift can actually disturb.
    """
    p, _ = generate_panel(skill=0.5)
    assert demeaned_label_autocorrelation(p) < label_autocorrelation(p)


def test_persistence_is_reported_but_does_not_gate_the_verdict():
    """The verdict must not depend on the persistence benchmark.

    That benchmark is attenuated by observation noise in the label, so gating on
    it produced false alarms on honest panels. It is retained as context only,
    and this test pins that decision down.
    """
    p, _ = generate_panel(skill=0.6)
    check = run_shift_test(p, offset=1)
    assert np.isfinite(check.detail["label_autocorrelation"])
    assert np.isfinite(check.detail["expected_under_persistence"])
    # Shifted IC sits above the naive persistence expectation, yet this is an
    # honest panel and must still pass.
    assert check.perturbed_ic > check.detail["expected_under_persistence"]
    assert check.passed is True


# ----------------------------------------------------------------------
# Suite behaviour
# ----------------------------------------------------------------------
def test_full_audit_clean_panel_has_no_failures():
    p, _ = generate_panel(skill=0.6)
    summary = alignment_summary(run_alignment_audit(p))
    assert summary["any_failed"] is False
    assert summary["all_passed"] is True


def test_full_audit_flags_the_misaligned_panel():
    summary = alignment_summary(run_alignment_audit(_misaligned_panel()))
    assert summary["any_failed"] is True
    assert "shift+1" in summary["failed"]


def test_inconclusive_is_distinct_from_passed():
    """A skill-free panel is not 'aligned', it is untestable; the summary must say so."""
    p, _ = generate_panel(skill=0.0)
    summary = alignment_summary(run_alignment_audit(p))
    assert summary["all_passed"] is False
    assert summary["any_failed"] is False
    assert summary["inconclusive"]
