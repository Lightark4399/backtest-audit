"""Tests for the survivorship audit.

The two directions that matter:

* **Detection** -- when attrition is coupled to predictability, dropping the
  leavers must inflate the score and the module must say so.
* **No false alarm** -- when attrition is *uncoupled*, the module must report no
  gap. Random attrition costs sample size but introduces no bias, and a check
  that flagged it would be flagging the mere fact that entities left, which is
  not a defect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit.audits.survivorship import (
    MATERIAL_GAP,
    delisted_entities,
    restrict_to_entities,
    run_survivorship_audit,
    surviving_entities,
)
from audit.coverage import AuditVerdict
from audit.panel import DATE, ENTITY, Panel
from audit.report.text import WIDTH, format_survivorship
from audit.run import run_baseline_audit
from audit.synthetic import generate_panel, generate_panel_with_delisting


# ----------------------------------------------------------------------
# Universe reconstruction mechanics
# ----------------------------------------------------------------------
def test_survivors_are_entities_present_on_the_final_date():
    dates = pd.bdate_range("2022-01-03", periods=4)
    frame = pd.DataFrame(
        {
            "entity_id": ["A"] * 4 + ["B"] * 2,  # B leaves after two dates
            "event_date": list(dates) + list(dates[:2]),
            "prediction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "label": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    p = Panel.from_frame(frame, train_end=dates[1])
    assert surviving_entities(p) == {"A"}
    assert delisted_entities(p) == {"B"}


def test_tail_dates_tolerates_a_single_missing_day():
    """An entity absent only on the final day is a data gap, not a delisting.

    Classifying it as a failure would overstate attrition and therefore overstate
    the bias -- a false alarm produced by the audit's own bookkeeping.
    """
    dates = pd.bdate_range("2022-01-03", periods=5)
    frame = pd.DataFrame(
        {
            "entity_id": ["A"] * 5 + ["B"] * 4,  # B misses only the last date
            "event_date": list(dates) + list(dates[:4]),
            "prediction": np.arange(9, dtype=float),
            "label": np.arange(9, dtype=float),
        }
    )
    p = Panel.from_frame(frame, train_end=dates[1])
    assert delisted_entities(p, tail_dates=1) == {"B"}
    assert delisted_entities(p, tail_dates=2) == set()


def test_restrict_preserves_train_boundary():
    p, _ = generate_panel_with_delisting()
    sub = restrict_to_entities(p, surviving_entities(p))
    assert sub.train_end == p.train_end
    assert sub.n_rows < p.n_rows


def test_restricting_to_nothing_raises():
    p, _ = generate_panel(skill=0.3)
    with pytest.raises(ValueError, match="no rows"):
        restrict_to_entities(p, set())


def test_delisting_metadata_is_consistent_with_the_panel():
    """Every entity with a delisting date must actually stop appearing."""
    p, meta = generate_panel_with_delisting()
    delisted = meta.loc[meta["delisting_date"].notna()]
    assert len(delisted) > 0
    for _, row in delisted.head(10).iterrows():
        rows = p.data.loc[p.data[ENTITY] == row["entity_id"]]
        assert rows[DATE].max() <= pd.Timestamp(row["delisting_date"])


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def test_attrition_coupled_to_predictability_inflates_the_score():
    """The failure mode: leavers were harder to forecast, so dropping them flatters."""
    p, _ = generate_panel_with_delisting(delist_hardness=0.0)
    res = run_survivorship_audit(p)
    assert res.n_entities_delisted > 0
    assert res.gap > MATERIAL_GAP
    assert res.passed is False
    assert res.survivors_demeaned_ic > res.pit_demeaned_ic


def test_uncoupled_attrition_produces_no_gap():
    """Entities leaving for reasons unrelated to the target is not a bias.

    ``delist_hardness=1.0`` removes the coupling: leavers are exactly as
    predictable as everyone else. The audit must report no material gap, or it
    would be flagging attrition itself rather than selection on outcome.
    """
    p, _ = generate_panel_with_delisting(delist_hardness=1.0)
    res = run_survivorship_audit(p)
    assert res.n_entities_delisted > 0  # attrition did happen
    assert abs(res.gap) < MATERIAL_GAP
    assert res.passed is True


def test_gap_grows_as_leavers_become_harder_to_predict():
    gaps = []
    for hardness in (1.0, 0.5, 0.0):
        p, _ = generate_panel_with_delisting(delist_hardness=hardness)
        gaps.append(run_survivorship_audit(p).gap)
    assert gaps == sorted(gaps), f"gap not monotone in coupling: {gaps}"
    assert gaps[-1] > gaps[0] + MATERIAL_GAP


def test_no_attrition_is_reported_as_untestable_not_as_a_pass():
    """A balanced panel cannot demonstrate the absence of survivorship bias.

    A universe that was assembled without its delisted entities in the first
    place looks exactly like this, and the absence is invisible from inside the
    data. Reporting PASS would certify something the data cannot show.
    """
    p, _ = generate_panel(skill=0.4)  # balanced: every entity on every date
    res = run_survivorship_audit(p)
    assert res.n_entities_delisted == 0
    assert res.passed is None
    assert "NO ATTRITION" in res.verdict
    assert "invisible" in res.verdict


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def test_result_serialises_with_the_counts_needed_to_interpret_it():
    p, _ = generate_panel_with_delisting(delist_hardness=0.0)
    d = run_survivorship_audit(p).to_dict()
    for key in ("gap", "n_entities_delisted", "survivor_rate", "passed", "verdict"):
        assert key in d
    assert 0.0 < d["survivor_rate"] < 1.0


def test_opposite_direction_is_reported_as_a_pass_with_an_explanation():
    """Leavers being EASIER to predict understates rather than flatters.

    Constructed directly rather than via the generator: doomed entities keep full
    skill while survivors are degraded, inverting the usual relationship. For a
    volatility-style target this is a plausible real pattern, so the module must
    handle it as a pass with an explanation rather than as an anomaly.
    """
    p, _ = generate_panel_with_delisting(delist_hardness=1.0)
    survivors = surviving_entities(p)
    data = p.data.copy()
    is_survivor = data[ENTITY].isin(survivors)
    level = data.groupby(ENTITY)["prediction"].transform("mean")
    data.loc[is_survivor, "prediction"] = level[is_survivor]  # survivors lose all skill
    degraded = Panel(data=data, train_end=p.train_end)

    res = run_survivorship_audit(degraded)
    assert res.gap < -MATERIAL_GAP
    assert res.passed is True
    assert "opposite direction" in res.verdict.lower()


def test_report_names_the_arm_as_the_panel_as_supplied():
    """The arm is the input panel, not a reconstructed point-in-time universe."""
    p, _ = generate_panel_with_delisting(delist_hardness=1.0)
    text = format_survivorship(run_survivorship_audit(p))
    assert "As-supplied panel (demeaned IC)" in text
    assert "Point-in-time universe" not in text


def test_undefined_gap_is_not_a_pass():
    """A constant prediction makes every demeaned IC undefined, so the gap is NaN.

    Old behaviour: the NaN fell through to PASS ("moves the demeaned IC by only
    +nan"). Target: passed=None with an INCONCLUSIVE verdict naming the
    undefined arm. Migration impact: such inputs now read INCONCLUSIVE, not PASS,
    in the result, the text and JSON reports, and the coverage manifest; every
    finite case is unchanged.
    """
    p, _ = generate_panel_with_delisting()
    constant = p.replace_prediction(pd.Series(1.0, index=p.data.index))
    res = run_survivorship_audit(constant)
    assert res.n_entities_delisted > 0
    assert not np.isfinite(res.gap)
    assert res.passed is None
    assert res.verdict.startswith("INCONCLUSIVE")
    assert "survivors-only and as-supplied demeaned IC could not be computed" in res.verdict
    assert "PASS" not in res.verdict and "nan" not in res.verdict

    audit = run_baseline_audit(constant)
    assert audit.to_dict()["survivorship"]["passed"] is None
    entry = audit.to_dict()["audit_coverage"]["audits"]["survivorship"]
    assert entry["verdict"] == AuditVerdict.INCONCLUSIVE.value
    text = format_survivorship(audit.survivorship)
    assert "[----]" in text and "[PASS]" not in text
    assert "nan" not in text.lower()
    for label in (
        "As-supplied panel (demeaned IC)",
        "Present in tail window (demeaned IC)",
        "Gap (tail-window subset minus as-supplied)",
    ):
        line = next(ln for ln in text.splitlines() if label in ln)
        assert line.rstrip().endswith("undefined"), line
    for line in text.splitlines():
        assert len(line) <= WIDTH, line

    # Precedence: a balanced panel with the same undefined scores is still NO
    # ATTRITION -- with nobody absent there is no survivors-only arm to compare.
    balanced, _ = generate_panel(skill=0.4)
    flat = balanced.replace_prediction(pd.Series(1.0, index=balanced.data.index))
    control = run_survivorship_audit(flat)
    assert control.passed is None
    assert control.verdict.startswith("NO ATTRITION")


def _survivors_degraded(p: Panel) -> Panel:
    survivors = surviving_entities(p)
    data = p.data.copy()
    is_survivor = data[ENTITY].isin(survivors)
    level = data.groupby(ENTITY)["prediction"].transform("mean")
    data.loc[is_survivor, "prediction"] = level[is_survivor]
    return Panel(data=data, train_end=p.train_end)


@pytest.mark.parametrize(
    ("case", "passed"),
    [("coupled", False), ("uncoupled", True), ("opposite", True)],
)
def test_finite_verdicts_report_the_measured_subset_gap_only(case, passed):
    """Every finite verdict states the final-date-subset vs as-supplied gap and
    does not infer why the absent entities score differently, that they were
    delisted, or that the sample was selected on outcome."""
    if case == "coupled":
        p, _ = generate_panel_with_delisting(delist_hardness=0.0)
    elif case == "uncoupled":
        p, _ = generate_panel_with_delisting(delist_hardness=1.0)
    else:
        p = _survivors_degraded(generate_panel_with_delisting(delist_hardness=1.0)[0])
    res = run_survivorship_audit(p)
    assert res.passed is passed
    assert "present on the final date" in res.verdict
    for claim in ("harder to predict", "easier to predict", "selected partly", "disappeared"):
        assert claim not in res.verdict
    if passed is False:
        assert "not verified delisting" in res.verdict

    text = format_survivorship(res)
    assert "Gap (tail-window subset minus as-supplied)" in text
    assert "attributable" not in text
    for line in text.splitlines():
        assert len(line) <= WIDTH, line


def test_tail_window_keeps_an_entity_missing_only_the_last_date():
    """tail_dates=2 keeps an entity absent on the last date but seen the date
    before; verdict and report must describe the window, not "the final date"."""
    p, _ = generate_panel_with_delisting(delist_hardness=0.0)
    last = p.dates[-1]
    gap_entity = sorted(surviving_entities(p))[0]
    data = p.data.loc[~((p.data[ENTITY] == gap_entity) & (p.data[DATE] == last))]
    holed = Panel(data=data.reset_index(drop=True), train_end=p.train_end)

    assert gap_entity not in surviving_entities(holed, tail_dates=1)
    assert gap_entity in surviving_entities(holed, tail_dates=2)

    res = run_survivorship_audit(holed, tail_dates=2)
    assert res.n_entities_surviving == len(surviving_entities(holed, tail_dates=2))
    assert "at least once in the final 2 dates" in res.verdict
    assert "on the final date" not in res.verdict

    text = format_survivorship(res)
    assert "absent from all of the final 2 dates" in text
    assert "Present in tail window (demeaned IC)" in text
    for line in text.splitlines():
        assert len(line) <= WIDTH, line


@pytest.mark.parametrize("tail_dates", [0, -1, "beyond"])
def test_invalid_tail_dates_raise_instead_of_misdescribing_the_window(tail_dates):
    """Old behaviour: the slice ``dates[-tail_dates:]`` silently selected every
    date (0, or N beyond the panel) or dropped the first dates (negative N),
    while the verdict and report said "the final N dates". Target: ValueError.
    Migration impact: only these invalid values change; 1 up to the panel's
    date count behave as before."""
    p, _ = generate_panel_with_delisting(delist_hardness=0.0)
    value = len(p.dates) + 1 if tail_dates == "beyond" else tail_dates
    with pytest.raises(ValueError, match="tail_dates must be between 1 and"):
        run_survivorship_audit(p, tail_dates=value)


def test_valid_tail_dates_are_unchanged():
    """Valid control: 1, 2 and the panel's full date count still run."""
    p, _ = generate_panel_with_delisting(delist_hardness=0.0)
    one = run_survivorship_audit(p, tail_dates=1)
    two = run_survivorship_audit(p, tail_dates=2)
    assert one.passed is False and one.n_entities_delisted > 0
    assert "on the final date" in one.verdict
    assert two.n_entities_surviving == len(surviving_entities(p, tail_dates=2))
    assert "at least once in the final 2 dates" in two.verdict
    full = run_survivorship_audit(p, tail_dates=len(p.dates))
    assert full.n_entities_delisted == 0
    assert full.verdict.startswith("NO ATTRITION")
