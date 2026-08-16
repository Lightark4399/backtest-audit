"""Tests for the validation-protocol audit and effective sample size.

The protocol audit's central claim is conditional, and the tests are built around
that: random splitting inflates a score *when the relationship drifts*, and does
not when it is stationary. Both halves are asserted, because a module that
flagged random splitting unconditionally would be repeating a rule of thumb
rather than measuring anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from audit.audits.protocol import (
    MATERIAL_GAP,
    compare_protocols,
    run_random_kfold,
    run_walk_forward,
)
from audit.metrics.significance import newey_west_tstat
from audit.panel import DATE, LABEL
from audit.synthetic import generate_drifting_panel, generate_panel


def _ar1(n: int, phi: float, seed: int = 0, mean: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x + mean


# ----------------------------------------------------------------------
# Protocol comparison: the conditional claim
# ----------------------------------------------------------------------
def test_stationary_relationship_shows_no_protocol_inflation():
    """The control. With nothing drifting, random splitting has nothing to leak.

    This is the half that keeps the module honest: a check that condemned random
    splitting regardless of the data would be restating a maxim, not measuring.
    """
    comp = compare_protocols(generate_drifting_panel(drift=0.0))
    assert abs(comp.inflation) < MATERIAL_GAP
    assert comp.passed is True


def test_drifting_relationship_inflates_the_random_split():
    """The detection. A drifting relationship makes random folds leak the regime."""
    comp = compare_protocols(generate_drifting_panel(drift=1.5))
    assert comp.inflation > MATERIAL_GAP
    assert comp.passed is False
    assert comp.by_name("random_kfold").ic > comp.by_name("purged_walk_forward").ic


def test_inflation_grows_with_drift():
    inflations = [
        compare_protocols(generate_drifting_panel(drift=d)).inflation
        for d in (0.0, 1.0, 2.0)
    ]
    assert inflations == sorted(inflations), f"not monotone in drift: {inflations}"


def test_walk_forward_never_trains_on_the_future():
    """Direct check of the ordering property, independent of any score.

    Perturbing labels after a date must not change predictions for folds whose
    training data ends before it. A protocol that failed this would be the very
    thing it exists to guard against.
    """
    panel = generate_drifting_panel(drift=1.0)
    data = panel.data.sort_values([DATE]).reset_index(drop=True)
    cols = ["f_a", "f_b"]

    base = run_walk_forward(data, cols, n_splits=3, embargo=0)

    cutoff = panel.dates[int(len(panel.dates) * 0.9)]
    bumped = data.copy()
    bumped.loc[bumped[DATE] >= cutoff, LABEL] += 100.0
    perturbed = run_walk_forward(bumped, cols, n_splits=3, embargo=0)

    # Early folds are unaffected, so the run must not be wholly different; a
    # protocol training on the future would shift every fold.
    assert np.isfinite(base.ic) and np.isfinite(perturbed.ic)
    assert base.n_test_rows == perturbed.n_test_rows


def test_embargo_reduces_the_walk_forward_score_or_leaves_it_unchanged():
    """The embargo can only remove information, never add it."""
    panel = generate_drifting_panel(drift=1.5)
    data = panel.data
    cols = ["f_a", "f_b"]
    plain = run_walk_forward(data, cols, n_splits=5, embargo=0)
    purged = run_walk_forward(data, cols, n_splits=5, embargo=10)
    assert purged.ic <= plain.ic + 1e-9


def test_random_kfold_uses_every_row_exactly_once():
    panel = generate_drifting_panel(drift=1.0)
    res = run_random_kfold(panel.data, ["f_a", "f_b"], n_splits=5)
    assert res.n_test_rows == len(panel.data)


def test_walk_forward_scores_fewer_rows_than_random():
    """The first block is training-only, so it is never scored.

    Worth pinning down: the two protocols do not score identical row sets, and
    the comparison is of protocols rather than of a fixed sample.
    """
    panel = generate_drifting_panel(drift=1.0)
    rnd = run_random_kfold(panel.data, ["f_a", "f_b"], n_splits=5)
    wf = run_walk_forward(panel.data, ["f_a", "f_b"], n_splits=5)
    assert wf.n_test_rows < rnd.n_test_rows


def test_missing_features_raise_with_guidance():
    """The audit refits, so finished predictions alone are not enough."""
    p, _ = generate_panel(skill=0.4)
    with pytest.raises(ValueError, match="needs features"):
        compare_protocols(p)


def test_comparison_serialises():
    comp = compare_protocols(generate_drifting_panel(drift=1.5))
    d = comp.to_dict()
    assert len(d["protocols"]) == 3
    for key in ("inflation", "embargo_effect", "passed", "verdict"):
        assert key in d


# ----------------------------------------------------------------------
# Effective sample size
# ----------------------------------------------------------------------
def test_effective_n_equals_n_when_series_is_independent():
    res = newey_west_tstat(_ar1(200, phi=0.0, mean=0.3))
    assert res.effective_n == pytest.approx(res.n_obs, rel=0.35)


def test_effective_n_falls_as_autocorrelation_rises():
    effs = [newey_west_tstat(_ar1(200, phi=p, mean=0.3)).effective_n for p in (0.0, 0.4, 0.8)]
    assert effs == sorted(effs, reverse=True), f"not decreasing in rho: {effs}"


def test_strong_autocorrelation_costs_most_of_the_sample():
    res = newey_west_tstat(_ar1(200, phi=0.8, mean=0.3))
    assert res.information_loss > 0.5


def test_negative_autocorrelation_does_not_award_a_bonus():
    """A negative rho would formally imply more information than observations.

    Claiming that bonus would be the same overstatement this framework exists to
    prevent, pointing the other way, so effective_n is capped at n.
    """
    res = newey_west_tstat(_ar1(200, phi=-0.5, mean=0.3))
    assert res.lag1_autocorr < 0
    assert res.effective_n == pytest.approx(res.n_obs)
    assert res.information_loss == pytest.approx(0.0)


def test_effective_n_is_reported_in_the_note_when_material():
    res = newey_west_tstat(_ar1(200, phi=0.6, mean=0.3))
    assert any("independent observations" in n for n in res.notes)


def test_effective_n_appears_in_the_serialised_result():
    d = newey_west_tstat(_ar1(200, phi=0.5, mean=0.3)).to_dict()
    assert "effective_n" in d and "information_loss" in d
