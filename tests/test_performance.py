"""Tests for the PnL layer.

The layer's job is to restate findings in Sharpe units without changing them. So
the tests check two things: that the mechanics are right, and that the raw and
demeaned figures reproduce the same conclusion the IC decomposition reaches. If
they disagreed, one of the two would be wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit.examples.pipelines import run_clean, run_leaky
from audit.metrics.ic import demeaned_ic, raw_ic
from audit.metrics.performance import (
    compare_performance,
    compute_pnl,
    max_drawdown,
    performance,
    rank_positions,
)
from audit.panel import DATE, LABEL, PRED, Panel
from audit.synthetic import generate_panel


# ----------------------------------------------------------------------
# Position construction
# ----------------------------------------------------------------------
def test_positions_are_dollar_neutral_and_unit_gross():
    p, _ = generate_panel(skill=0.5)
    view = p.test_slice().copy()
    view["position"] = rank_positions(view)

    for _, g in view.groupby(DATE):
        pos = g["position"].to_numpy()
        assert abs(pos.sum()) < 1e-9, "book must be dollar-neutral"
        assert abs(np.abs(pos).sum() - 1.0) < 1e-9, "gross exposure must be 1"


def test_positions_depend_only_on_ordering():
    """Rank-based weights must be invariant to a monotone rescaling.

    This is what makes the PnL the same information as a rank IC rather than a
    second, different test that happens to run alongside it.
    """
    p, _ = generate_panel(skill=0.5)
    view = p.test_slice().copy()
    a = rank_positions(view)

    rescaled = view.copy()
    rescaled[PRED] = np.exp(rescaled[PRED])  # strictly increasing
    b = rank_positions(rescaled)

    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)


def test_constant_predictions_produce_no_position():
    """With no ordering to act on, taking a position would report a tie-break."""
    p, _ = generate_panel(skill=0.5)
    view = p.test_slice().copy()
    view[PRED] = 3.0
    assert (rank_positions(view) == 0.0).all()


def test_single_entity_date_produces_no_position():
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "A", "B"],
            "event_date": ["2022-01-03", "2022-01-04", "2022-01-04"],
            "prediction": [1.0, 1.0, 2.0],
            "label": [0.1, 0.2, 0.3],
        }
    )
    p = Panel.from_frame(frame, train_end="2022-01-03")
    view = p.data
    pos = rank_positions(view)
    # The single-entity date cannot support a neutral book
    solo = view[DATE] == pd.Timestamp("2022-01-03")
    assert (pos[solo] == 0.0).all()


# ----------------------------------------------------------------------
# PnL and statistics
# ----------------------------------------------------------------------
def test_pnl_is_positions_times_labels():
    """Hand-computable case, so the arithmetic is pinned rather than assumed."""
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "B", "C", "D"],
            "event_date": ["2022-01-03"] * 4,
            "prediction": [1.0, 2.0, 3.0, 4.0],
            "label": [10.0, 20.0, 30.0, 40.0],
        }
    )
    p = Panel.from_frame(frame, train_end="2021-12-31")
    table = compute_pnl(p, scope="all")

    # Ranks 1,2,3,4 → centred −1.5,−0.5,0.5,1.5 → gross 4 → weights ±0.375, ±0.125
    expected = -0.375 * 10 - 0.125 * 20 + 0.125 * 30 + 0.375 * 40
    assert float(table["pnl"].iloc[0]) == pytest.approx(expected)


def test_perfect_prediction_beats_random_prediction():
    p, _ = generate_panel(skill=0.5)
    perfect = p.replace_prediction(p.data[LABEL])
    rng = np.random.default_rng(0)
    noise = p.replace_prediction(pd.Series(rng.normal(size=p.n_rows), index=p.data.index))

    assert performance(perfect).sharpe > performance(noise).sharpe


def test_sign_flipped_prediction_reverses_pnl():
    p, _ = generate_panel(skill=0.6)
    flipped = p.replace_prediction(-p.data[PRED])
    a = performance(p, demean_labels=True)
    b = performance(flipped, demean_labels=True)
    assert np.sign(a.mean_return) != np.sign(b.mean_return)


def test_max_drawdown_is_zero_for_a_monotone_curve():
    assert max_drawdown(pd.Series([1.0, 1.1, 1.2, 1.5])) == pytest.approx(0.0)


def test_max_drawdown_measures_peak_to_trough():
    curve = pd.Series([1.0, 2.0, 1.0, 3.0])
    assert max_drawdown(curve) == pytest.approx(0.5)


def test_turnover_is_reported():
    p, _ = generate_panel(skill=0.5)
    stats = performance(p)
    assert stats.turnover > 0


def test_short_series_refuses_to_summarise():
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "B"],
            "event_date": ["2022-01-03"] * 2,
            "prediction": [1.0, 2.0],
            "label": [0.1, 0.2],
        }
    )
    p = Panel.from_frame(frame, train_end="2021-12-31")
    with pytest.raises(ValueError, match="at least two dates"):
        performance(p, scope="all")


# ----------------------------------------------------------------------
# The finding: raw Sharpe reproduces the level effect
# ----------------------------------------------------------------------
def test_zero_skill_shows_an_absurd_raw_sharpe_and_a_nil_demeaned_one():
    """The PnL layer must reproduce the IC decomposition's conclusion.

    A prediction knowing only the entity level wins every day on a sign-constant
    target, because the book is long the persistently-high names. If the layer
    reported only the raw figure it would be manufacturing the deception the
    framework exists to expose.
    """
    p, _ = generate_panel(skill=0.0, level_leak=1.0)

    raw = performance(p, demean_labels=False)
    dm = performance(p, demean_labels=True)

    assert raw.sharpe_annualised > 20, "level alone should look spectacular"
    assert raw.hit_rate > 0.95
    assert raw.max_drawdown == pytest.approx(0.0)

    assert dm.sharpe_annualised < 10, "no skill should survive demeaning"
    assert dm.hit_rate < 0.7

    # And it agrees with what the IC decomposition says about the same panel
    assert raw_ic(p).mean > 0.4
    assert abs(demeaned_ic(p).mean) < 0.05


def test_demeaned_sharpe_ranks_panels_like_demeaned_ic():
    """The two metrics must order the same panels the same way.

    They are meant to be the same information in different units; disagreement
    would mean one of them is measuring something else.
    """
    panels = {
        "level_only": generate_panel(skill=0.0)[0],
        "some_skill": generate_panel(skill=0.3)[0],
        "more_skill": generate_panel(skill=0.6)[0],
    }
    by_sharpe = sorted(panels, key=lambda k: performance(panels[k], demean_labels=True).sharpe)
    by_ic = sorted(panels, key=lambda k: demeaned_ic(panels[k]).mean)
    assert by_sharpe == by_ic


def test_compare_performance_reports_both_columns():
    table = compare_performance({"clean": run_clean(), "leaky": run_leaky()})
    assert {"sharpe_raw", "sharpe_demeaned"} <= set(table.columns)
    assert table.loc["leaky", "sharpe_demeaned"] > table.loc["clean", "sharpe_demeaned"]


def test_hac_statistic_is_reported_alongside_the_sharpe():
    """Annualising by sqrt(252) assumes independence; the report must not hide that."""
    p, _ = generate_panel(skill=0.5)
    stats = performance(p, demean_labels=True)
    assert np.isfinite(stats.sharpe_tstat)
    assert np.isfinite(stats.sharpe_tstat_hac)
    assert "effective_n" in stats.to_dict()
