"""Tests for the bitemporal store and the point-in-time audit.

Two layers, as elsewhere: the SQL is checked for the properties that make it
correct, and the audit is checked against scenarios whose answer is known because
the revision history was constructed rather than observed.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from audit.audits.pit import MATERIAL_GAP, run_pit_audit
from audit.ingest.duckdb_store import BitemporalStore, RevisionSpec
from audit.synthetic import generate_panel


@pytest.fixture(scope="module")
def observations() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    p, _ = generate_panel(skill=0.4)
    obs = p.data[["entity_id", "event_date", "label"]].rename(columns={"label": "value"})
    return obs, obs.copy(), p.train_end


# ----------------------------------------------------------------------
# Bitemporal storage
# ----------------------------------------------------------------------
def test_revision_is_stored_as_a_new_row_not_an_update():
    """The original belief must survive the correction.

    If a revision overwrote the first value, the historical record would become
    the current view of the past and no as-of query could reconstruct what was
    actually known. Storing both versions is what makes the audit possible at
    all.
    """
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_date": ["2022-01-03", "2022-01-04"],
            "value": [1.0, 2.0],
        }
    )
    with BitemporalStore() as store:
        store.load_observations(frame, revisions=RevisionSpec(fraction=1.0, lag_days=3))
        raw = store.con.execute(
            "SELECT * FROM observation_raw ORDER BY event_date, knowledge_date"
        ).df()
        # Two dates, each with an initial and a corrected version
        assert len(raw) == 4
        assert set(raw["source"]) == {"initial", "revision"}


def test_asof_returns_the_pre_correction_value_before_the_correction_lands():
    """The heart of the audit: as-of must not see a future correction."""
    frame = pd.DataFrame(
        {"entity_id": ["A"], "event_date": ["2022-01-03"], "value": [10.0]}
    )
    with BitemporalStore() as store:
        store.load_observations(frame, revisions=RevisionSpec(fraction=1.0, lag_days=5, seed=1))

        raw = store.con.execute(
            "SELECT value, knowledge_date FROM observation_raw ORDER BY knowledge_date"
        ).df()
        initial_value, final_value = float(raw["value"].iloc[0]), float(raw["value"].iloc[1])
        assert initial_value != final_value  # the scenario actually revised something

        # Before the correction arrives, the original value is what was known
        before = store.asof("2022-01-05")
        assert float(before["value"].iloc[0]) == pytest.approx(initial_value)

        # After it arrives, the corrected value
        after = store.asof("2022-01-10")
        assert float(after["value"].iloc[0]) == pytest.approx(final_value)

        # And the restated view always shows the correction, whatever the date
        assert float(store.restated()["value"].iloc[0]) == pytest.approx(final_value)


def test_asof_excludes_dates_that_had_not_happened_yet():
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "A", "A"],
            "event_date": ["2022-01-03", "2022-01-10", "2022-01-17"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    with BitemporalStore() as store:
        store.load_observations(frame)
        got = store.asof("2022-01-11")
        assert len(got) == 2  # the 17th has not occurred


def test_knowledge_date_before_event_date_is_rejected():
    """A value cannot be known before the day it describes."""
    with BitemporalStore() as store:
        with pytest.raises(duckdb.ConstraintException):
            store.con.execute(
                "INSERT INTO observation_raw VALUES "
                "('A', DATE '2022-01-10', DATE '2022-01-05', 1.0, 'bad', NULL)"
            )


def test_universe_includes_delisted_entities_for_dates_they_traded():
    """Survivorship-bias-free membership is by date, not by present existence."""
    ents = pd.DataFrame(
        {
            "entity_id": ["ALIVE", "DEAD"],
            "sector": ["x", "x"],
            "listing_date": ["2020-01-01", "2020-01-01"],
            "delisting_date": [None, "2022-06-30"],
        }
    )
    with BitemporalStore() as store:
        store.load_entities(ents)
        assert store.universe("2022-01-01") == ["ALIVE", "DEAD"]
        assert store.universe("2022-12-31") == ["ALIVE"]


def test_features_use_a_strictly_past_window():
    """Hand-computable fixture: the trailing mean must exclude the current row.

    Values 1..5 on consecutive dates give f_mean5 of NULL, 1, 1.5, 2, 2.5. If the
    frame included the current row the sequence would be 1, 1.5, 2, 2.5, 3 --
    every value shifted and the first not null.
    """
    dates = pd.bdate_range("2022-01-03", periods=5)
    frame = pd.DataFrame(
        {
            "entity_id": ["A"] * 5,
            "event_date": dates,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    with BitemporalStore() as store:
        store.load_observations(frame)
        f = store.features("restated").sort_values("event_date")
        got = [None if pd.isna(v) else float(v) for v in f["f_mean5"]]
        assert got == [None, 1.0, 1.5, 2.0, 2.5]
        lag = [None if pd.isna(v) else float(v) for v in f["f_lag1"]]
        assert lag == [None, 1.0, 2.0, 3.0, 4.0]


def test_revision_summary_reports_size_and_lag():
    frame = pd.DataFrame(
        {
            "entity_id": ["A"] * 3,
            "event_date": pd.bdate_range("2022-01-03", periods=3),
            "value": [1.0, 2.0, 3.0],
        }
    )
    with BitemporalStore() as store:
        store.load_observations(frame, revisions=RevisionSpec(fraction=1.0, lag_days=7, seed=5))
        rev = store.revision_summary()
        assert len(rev) == 3
        assert (rev["revision_lag_days"] == 7).all()
        assert (rev["revision_size"].abs() > 0).all()


# ----------------------------------------------------------------------
# The audit
# ----------------------------------------------------------------------
def test_no_revisions_gives_no_gap(observations):
    """Control: with nothing corrected the two vintages must coincide.

    A comparison that reported a gap here would be measuring its own machinery.
    """
    obs, lab, train_end = observations
    res = run_pit_audit(obs, lab, train_end=train_end, revisions=None, max_asof_dates=12)
    assert res.n_revisions == 0
    assert abs(res.gap) < MATERIAL_GAP
    assert res.passed is None  # "no revisions" is not a pass; there was nothing to test
    assert "NO REVISIONS" in res.verdict


def test_revisions_that_correct_toward_truth_inflate_the_restated_score(observations):
    """The failure mode the module exists to detect."""
    obs, lab, train_end = observations
    res = run_pit_audit(
        obs, lab, train_end=train_end, revisions=RevisionSpec(fraction=0.3), max_asof_dates=12
    )
    assert res.n_revisions > 0
    assert res.gap > MATERIAL_GAP
    assert res.passed is False
    assert res.restated_demeaned_ic > res.asof_demeaned_ic


def test_gap_grows_with_the_revision_rate(observations):
    """Dose-response: more corrections must mean more unearned advantage.

    A module that flagged any revision scenario equally could be responding to
    the presence of the scenario rather than to its severity.
    """
    obs, lab, train_end = observations
    gaps = []
    for fraction in (0.1, 0.35, 0.6):
        res = run_pit_audit(
            obs,
            lab,
            train_end=train_end,
            revisions=RevisionSpec(fraction=fraction),
            max_asof_dates=10,
        )
        gaps.append(res.gap)
    assert gaps == sorted(gaps), f"gap not monotone in revision rate: {gaps}"
    assert gaps[-1] > 2 * gaps[0]


def test_restated_score_is_unaffected_by_the_revision_scenario(observations):
    """The restated arm sees final values, so it must not move with the lag or rate.

    This isolates the gap: if the restated score also moved, the comparison would
    be confounded and the difference could not be attributed to vintage.
    """
    obs, lab, train_end = observations
    a = run_pit_audit(
        obs, lab, train_end=train_end, revisions=RevisionSpec(fraction=0.2), max_asof_dates=8
    )
    b = run_pit_audit(
        obs, lab, train_end=train_end, revisions=RevisionSpec(fraction=0.6), max_asof_dates=8
    )
    assert a.restated_demeaned_ic == pytest.approx(b.restated_demeaned_ic, abs=1e-9)


def test_verdict_declines_to_call_no_revisions_a_clean_bill(observations):
    """Absence of this leak is not absence of leakage, and the report must say so."""
    obs, lab, train_end = observations
    res = run_pit_audit(obs, lab, train_end=train_end, revisions=None, max_asof_dates=8)
    assert "not a clean bill of health" in res.verdict


def test_result_serialises_for_ci_assertions(observations):
    obs, lab, train_end = observations
    res = run_pit_audit(
        obs, lab, train_end=train_end, revisions=RevisionSpec(fraction=0.3), max_asof_dates=8
    )
    d = res.to_dict()
    for key in ("gap", "restated_demeaned_ic", "asof_demeaned_ic", "revision_rate", "passed"):
        assert key in d
    assert np.isfinite(d["gap"])
