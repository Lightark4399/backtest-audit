"""Tests for the execution-timing audit.

Three profiles must be distinguished, and the third is the one a naive
implementation gets wrong: a signal with no edge on the contemporaneous return
and a real edge one period later is the *healthiest* possible result, not a
failure to decay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit.audits.execution import (
    COLLAPSE_RATIO,
    audit_execution_timing,
    shift_returns,
)
from audit.panel import ENTITY, Panel
from audit.synthetic import generate_panel, generate_return_panel


# ----------------------------------------------------------------------
# Shifting mechanics
# ----------------------------------------------------------------------
def test_shift_pairs_the_prediction_with_the_intended_return():
    dates = pd.bdate_range("2022-01-03", periods=4)
    frame = pd.DataFrame(
        {
            "entity_id": ["A"] * 4,
            "event_date": dates,
            "prediction": [0.0] * 4,
            "label": [0.0] * 4,
            "forward_return": [1.0, 2.0, 3.0, 4.0],
        }
    )
    p = Panel.from_frame(frame, train_end=dates[1])

    at0 = shift_returns(p, "forward_return", 0)
    assert list(at0["_ret"]) == [1.0, 2.0, 3.0, 4.0]

    at1 = shift_returns(p, "forward_return", 1)
    assert list(at1["_ret"]) == [2.0, 3.0, 4.0]  # last row has no counterpart


def test_shift_does_not_borrow_across_entities():
    dates = pd.bdate_range("2022-01-03", periods=3)
    frame = pd.DataFrame(
        {
            "entity_id": ["A"] * 3 + ["B"] * 3,
            "event_date": list(dates) * 2,
            "prediction": [0.0] * 6,
            "label": [0.0] * 6,
            "forward_return": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        }
    )
    p = Panel.from_frame(frame, train_end=dates[0])
    shifted = shift_returns(p, "forward_return", 1)
    a_rows = shifted.loc[shifted[ENTITY] == "A", "_ret"]
    assert 100.0 not in set(a_rows)


def test_rows_without_a_counterpart_are_dropped_not_filled():
    """A fabricated return would make the verdict depend on the fill rule."""
    p = generate_return_panel()
    at0 = shift_returns(p, "forward_return", 0)
    at2 = shift_returns(p, "forward_return", 2)
    assert len(at2) < len(at0)
    assert at2["_ret"].notna().all()


# ----------------------------------------------------------------------
# The three profiles
# ----------------------------------------------------------------------
def test_lookahead_signal_collapses_at_lag_one():
    """Profitable at lag 0 and flat at lag 1 is the signature of reading, not forecasting."""
    res = audit_execution_timing(generate_return_panel(lookahead=1.0))
    assert res.at_lag(0).ic > 0.5
    assert res.at_lag(1).ic < 0.1
    assert res.decay_ratio < COLLAPSE_RATIO
    assert res.passed is False


def test_honest_forecast_has_no_edge_on_the_contemporaneous_return():
    """The healthiest profile, and the one a naive decay test misreads.

    The lag-0 return had already happened when the signal was computed, so a
    genuine forecast should have no relationship with it. Ability appearing only
    at lag 1 is what forecasting looks like — and an earlier version of this
    module reported it as INCONCLUSIVE.
    """
    res = audit_execution_timing(generate_return_panel(lookahead=0.0))
    assert abs(res.at_lag(0).ic) < 0.05
    assert res.at_lag(1).ic > 0.05
    assert res.passed is True
    assert "clean forecast profile" in res.verdict


def test_partial_lookahead_is_also_caught():
    """A signal half-built from the contemporaneous return still fails."""
    res = audit_execution_timing(generate_return_panel(lookahead=0.5))
    assert res.passed is False
    assert res.at_lag(0).ic > res.at_lag(1).ic


def test_decay_ratio_falls_as_lookahead_rises():
    ratios = [
        audit_execution_timing(generate_return_panel(lookahead=la)).decay_ratio
        for la in (0.5, 0.75, 1.0)
    ]
    assert ratios == sorted(ratios, reverse=True), f"not monotone: {ratios}"


def test_signal_with_no_edge_at_any_lag_is_inconclusive():
    """Nothing to read a profile from, so no verdict is asserted."""
    p = generate_return_panel()
    rng = np.random.default_rng(3)
    noise = p.replace_prediction(pd.Series(rng.normal(size=p.n_rows), index=p.data.index))
    # replace_prediction keeps the other columns, including forward_return
    res = audit_execution_timing(noise)
    assert res.passed is None
    assert "INCONCLUSIVE" in res.verdict


# ----------------------------------------------------------------------
# Requirements and reporting
# ----------------------------------------------------------------------
def test_panel_without_returns_raises_with_guidance():
    """Delaying execution against a non-tradeable target measures nothing."""
    p, _ = generate_panel(skill=0.4)
    with pytest.raises(ValueError, match="tradeable"):
        audit_execution_timing(p)


def test_sharpe_is_reported_at_every_lag():
    res = audit_execution_timing(generate_return_panel(lookahead=1.0))
    for r in res.results:
        assert "sharpe" in r.to_dict()


def test_result_serialises():
    d = audit_execution_timing(generate_return_panel(lookahead=1.0)).to_dict()
    assert len(d["lags"]) == 3
    for key in ("decay_ratio", "passed", "verdict", "return_column"):
        assert key in d


def test_lag_zero_scores_more_rows_than_lag_two():
    """Later lags lose the tail of each entity's history, and that must be visible."""
    res = audit_execution_timing(generate_return_panel(lookahead=1.0))
    assert res.at_lag(0).n_rows > res.at_lag(2).n_rows
