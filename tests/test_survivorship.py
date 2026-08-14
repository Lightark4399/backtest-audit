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
from audit.panel import DATE, ENTITY, Panel
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
