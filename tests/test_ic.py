"""Correctness tests for the IC layer.

Every assertion here is against a case whose answer is known by construction.
Tests that merely compare today's output to a stored value from yesterday would
lock in whatever bugs exist today, so they are avoided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit.metrics.ic import (
    MIN_CROSS_SECTION,
    cross_sectional_ic,
    demeaned_ic,
    rank_ic,
    raw_ic,
)
from audit.panel import Panel, PanelError
from audit.synthetic import (
    generate_panel,
    generate_perfect_panel,
    generate_random_panel,
)


# ----------------------------------------------------------------------
# Boundary behaviour: the values an IC must take when the answer is certain
# ----------------------------------------------------------------------
def test_perfect_prediction_gives_ic_one():
    p = generate_perfect_panel()
    assert raw_ic(p).mean == pytest.approx(1.0, abs=1e-12)
    assert rank_ic(p).mean == pytest.approx(1.0, abs=1e-12)


def test_independent_prediction_gives_ic_near_zero():
    p = generate_random_panel()
    # With ~60 entities per date over ~60 evaluation dates the standard error of
    # the mean daily IC is roughly 1/sqrt(60*60); 0.05 is a loose but safe bound.
    assert abs(raw_ic(p).mean) < 0.05
    assert abs(rank_ic(p).mean) < 0.05


def test_sign_flipped_prediction_gives_ic_minus_one():
    p = generate_perfect_panel()
    flipped = p.replace_prediction(-p.data["label"])
    assert raw_ic(flipped).mean == pytest.approx(-1.0, abs=1e-12)


def test_rank_ic_invariant_under_monotone_transform():
    """Spearman must not change when the prediction is monotonically rescaled.

    This distinguishes a genuine rank statistic from a Pearson correlation on
    raw values, and would catch an implementation that forgot to rank.
    """
    p, _ = generate_panel(skill=0.5)
    base = rank_ic(p).mean
    # exp() is strictly increasing, so all within-date orderings are preserved
    transformed = p.replace_prediction(np.exp(p.data["prediction"]))
    assert rank_ic(transformed).mean == pytest.approx(base, abs=1e-10)


# ----------------------------------------------------------------------
# Degenerate cross-sections must be reported, not silently coerced
# ----------------------------------------------------------------------
def test_constant_prediction_yields_undefined_not_zero():
    """A prediction constant within a date carries no ranking information.

    Its correlation is undefined (zero variance), and reporting 0.0 would imply
    "measured, and found to be no better than chance" when the truth is "not
    measurable". The date must be excluded and counted.
    """
    p = generate_random_panel(n_entities=20, n_dates=10)
    const = p.replace_prediction(pd.Series(5.0, index=p.data.index))
    series = raw_ic(const)
    assert series.n_dates_used == 0
    assert series.n_dates_undefined == series.n_dates_total
    assert np.isnan(series.mean)


def test_cross_sections_below_minimum_size_are_skipped():
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "B", "A", "B", "C"],
            "event_date": ["2022-01-03"] * 2 + ["2022-01-04"] * 3,
            "prediction": [1.0, 2.0, 1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 1.0, 2.0, 3.0],
        }
    )
    p = Panel.from_frame(frame, train_end="2022-01-03")
    series = cross_sectional_ic(p, scope="all")
    # The 2-entity date is below MIN_CROSS_SECTION and must be skipped
    assert MIN_CROSS_SECTION == 3
    assert series.n_dates_too_small == 1
    assert series.n_dates_used == 1


# ----------------------------------------------------------------------
# The central claim of the framework
# ----------------------------------------------------------------------
def test_zero_skill_with_level_knowledge_has_high_raw_ic_but_zero_demeaned_ic():
    """The failure mode this project exists to detect.

    A prediction containing the entity level and nothing else scores highly on
    raw IC while carrying no information about dynamics. Demeaned IC must expose
    that.
    """
    p, _ = generate_panel(skill=0.0, level_leak=1.0)
    assert raw_ic(p).mean > 0.4, "level knowledge alone should inflate raw IC"
    assert abs(demeaned_ic(p).mean) < 0.05, "demeaned IC must not credit level knowledge"


def test_demeaned_ic_increases_with_true_skill():
    """Monotonicity in the generative skill parameter.

    A metric that is high for the wrong reasons could still pass the zero-skill
    test by accident; requiring monotone response to genuine skill pins down
    that it measures the intended quantity.
    """
    means = []
    for skill in (0.0, 0.25, 0.5, 0.75):
        p, _ = generate_panel(skill=skill)
        means.append(demeaned_ic(p).mean)
    assert means == sorted(means), f"not monotone in skill: {means}"
    assert means[-1] > 0.4


def test_demeaned_ic_requires_train_end():
    """Refusing to compute is correct when the training boundary is unknown.

    Falling back to a full-sample mean would demean the labels using information
    from the evaluation period -- leakage committed by the audit tool itself.
    """
    p, _ = generate_panel()
    no_split = Panel(data=p.data, train_end=None)
    with pytest.raises(PanelError, match="train_end"):
        demeaned_ic(no_split)


def test_demeaning_uses_training_mean_only():
    """Verify directly that the evaluation period does not affect the mean used.

    Constructed so that the entity's test-period labels are wildly different from
    its training labels: if the implementation used a full-sample mean, the
    demeaned values would change when the test labels change.
    """
    dates = pd.bdate_range("2022-01-03", periods=8)
    rows = []
    for e in ("A", "B", "C", "D"):
        for i, d in enumerate(dates):
            rows.append(
                {"entity_id": e, "event_date": d, "prediction": float(i), "label": float(i)}
            )
    frame = pd.DataFrame(rows)
    train_end = dates[3]

    p1 = Panel.from_frame(frame, train_end=train_end)
    mu1 = p1.per_entity_train_mean("label")

    # Perturb only the test-period labels
    frame2 = frame.copy()
    mask = frame2["event_date"] > train_end
    frame2.loc[mask, "label"] += 1000.0
    p2 = Panel.from_frame(frame2, train_end=train_end)
    mu2 = p2.per_entity_train_mean("label")

    pd.testing.assert_series_equal(mu1, mu2)


# ----------------------------------------------------------------------
# Panel contract
# ----------------------------------------------------------------------
def test_duplicate_entity_date_rejected():
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_date": ["2022-01-03", "2022-01-03"],
            "prediction": [1.0, 2.0],
            "label": [1.0, 2.0],
        }
    )
    with pytest.raises(PanelError, match="duplicate"):
        Panel.from_frame(frame)


def test_missing_column_rejected():
    frame = pd.DataFrame({"entity_id": ["A"], "event_date": ["2022-01-03"], "label": [1.0]})
    with pytest.raises(PanelError, match="missing required column"):
        Panel.from_frame(frame)


def test_incomplete_rows_counted_when_dropped():
    frame = pd.DataFrame(
        {
            "entity_id": ["A", "B", "C", "D"],
            "event_date": ["2022-01-03"] * 4,
            "prediction": [1.0, 2.0, np.nan, 4.0],
            "label": [1.0, 2.0, 3.0, np.nan],
        }
    )
    p = Panel.from_frame(frame, train_end="2022-01-03")
    assert p.n_dropped == 2
    assert p.n_rows == 2
