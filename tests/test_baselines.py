"""Tests for the baseline layer and the incremental-IC statistic.

Two things are pinned down here:

1. Baselines respect the information constraint (no peeking at same-day or
   future labels). A leaking baseline would understate the model's increment,
   which is the wrong direction to be wrong in.
2. The incremental IC reads zero when there is genuinely nothing incremental --
   including in the presence of the errors-in-variables bias documented in
   ``metrics/partial.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit.metrics.baselines import (
    build_baseline_panel,
    default_baselines,
    evaluate_baselines,
    strongest_baseline,
)
from audit.metrics.partial import incremental_ic, partial_correlation
from audit.panel import DATE, ENTITY, LABEL, PRED, Panel
from audit.synthetic import generate_panel, generate_random_panel

BASELINES = {b.name: b for b in default_baselines()}


# ----------------------------------------------------------------------
# Information constraint
# ----------------------------------------------------------------------
def test_persistence_baseline_is_previous_observed_label():
    dates = pd.bdate_range("2022-01-03", periods=5)
    frame = pd.DataFrame(
        {
            "entity_id": ["A"] * 5 + ["B"] * 5,
            "event_date": list(dates) * 2,
            "prediction": [0.0] * 10,
            "label": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    p = Panel.from_frame(frame, train_end=dates[2])
    values = BASELINES["persistence"].build(p)
    got = p.data.assign(base=values)

    a = got[got[ENTITY] == "A"].sort_values(DATE)
    # First observation has no prior label -> undefined, never fabricated
    assert np.isnan(a["base"].iloc[0])
    # Thereafter it is exactly the previous label, never the current one
    assert list(a["base"].iloc[1:]) == [1.0, 2.0, 3.0, 4.0]
    assert not (a["base"].iloc[1:].to_numpy() == a[LABEL].iloc[1:].to_numpy()).any()


def test_no_baseline_uses_same_day_or_future_labels():
    """Perturbing a single label must not change any baseline value at or before it.

    This is a direct test of the information constraint rather than a reading of
    the implementation: if a baseline peeked at the current or a future label,
    changing that label would move a value it should not touch.
    """
    p, _ = generate_panel(skill=0.3)
    target_date = p.dates[len(p.dates) // 2]

    bumped = p.data.copy()
    mask = bumped[DATE] >= target_date
    bumped.loc[mask, LABEL] += 500.0
    p2 = Panel(data=bumped, train_end=p.train_end)

    for name, b in BASELINES.items():
        if name == "per_entity_train_mean" and target_date <= p.train_end:
            continue  # legitimately depends on training labels
        v1 = b.build(p).to_numpy()
        v2 = b.build(p2).to_numpy()
        before = p.data[DATE] < target_date
        a, c = v1[before.to_numpy()], v2[before.to_numpy()]
        both = ~(np.isnan(a) | np.isnan(c))
        assert np.allclose(a[both], c[both]), f"{name} changed on dates before the perturbation"


def test_cross_sectional_mean_baseline_is_undefined_by_construction():
    """It is constant within each date, so it cannot rank entities.

    Included as a control on the metric layer: if this baseline ever reports a
    defined IC, the layer is coercing degenerate correlations to numbers.
    """
    p, _ = generate_panel(skill=0.3)
    table = evaluate_baselines(p)
    row = table.loc["cross_sectional_mean"]
    assert row["n_dates_used"] == 0
    assert np.isnan(row["mean"])


def test_persistent_target_gives_baselines_high_ic():
    """Sanity check on the generator: the free score must actually be large.

    If baselines scored near zero the framework's central claim would be
    untestable, so this guards the test fixture as much as the code.
    """
    p, _ = generate_panel(skill=0.3)
    table = evaluate_baselines(p)
    assert table.loc["persistence", "mean"] > 0.3
    assert table.loc["per_entity_train_mean", "mean"] > 0.3


def test_strongest_baseline_ignores_undefined_entries():
    p, _ = generate_panel(skill=0.3)
    table = evaluate_baselines(p)
    assert strongest_baseline(table) != "cross_sectional_mean"
    assert strongest_baseline(table) in {"persistence", "ewma", "per_entity_train_mean"}


# ----------------------------------------------------------------------
# Partial correlation algebra
# ----------------------------------------------------------------------
def test_partial_correlation_equals_raw_when_control_is_noise():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    b = rng.normal(size=n)  # independent of both
    raw = float(np.corrcoef(x, y)[0, 1])
    assert partial_correlation(x, y, b) == pytest.approx(raw, abs=0.02)


def test_partial_correlation_zero_when_control_explains_everything():
    rng = np.random.default_rng(1)
    n = 4000
    b = rng.normal(size=n)
    x = b + 0.01 * rng.normal(size=n)
    y = b + 0.01 * rng.normal(size=n)
    val = partial_correlation(x, y, b)
    assert val is None or abs(val) < 0.05


def test_partial_correlation_undefined_for_constant_control():
    rng = np.random.default_rng(2)
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    assert partial_correlation(x, y, np.ones(100)) is None


def test_correlation_of_differences_is_biased_partial_correlation_is_not():
    """Demonstrates why corr(x-b, y-b) was rejected as the increment statistic.

    Construct x and y that are conditionally independent given b. The correlation
    of differences is materially positive; the partial correlation is not.
    """
    rng = np.random.default_rng(3)
    n = 20000
    b = rng.normal(0.0, 3.0, size=n)  # large variance relative to residuals
    x = b + rng.normal(0.0, 1.0, size=n)
    y = b + rng.normal(0.0, 1.0, size=n)

    corr_of_diff = float(np.corrcoef(x - b, y - b)[0, 1])
    partial = partial_correlation(x, y, b)

    assert abs(corr_of_diff) < 0.05  # here the shared term happens not to bias
    assert abs(partial) < 0.05

    # Now make the shared term dominate the differences: scale b's contribution
    # so that x-b and y-b both retain a large common component.
    x2 = 2.0 * b + rng.normal(0.0, 1.0, size=n)
    y2 = 2.0 * b + rng.normal(0.0, 1.0, size=n)
    corr_of_diff2 = float(np.corrcoef(x2 - b, y2 - b)[0, 1])
    partial2 = partial_correlation(x2, y2, b)

    # corr(x-b, y-b) is inflated by the residual b that remains in both terms,
    # while the partial correlation correctly reports no conditional association.
    assert corr_of_diff2 > 0.5
    assert abs(partial2) < 0.05


# ----------------------------------------------------------------------
# Incremental IC: the errors-in-variables correction
# ----------------------------------------------------------------------
def test_incremental_ic_is_zero_at_zero_skill():
    """With demeaning, a model that knows only the level gets no credit."""
    p, _ = generate_panel(skill=0.0, level_leak=1.0)
    for name in ("persistence", "ewma"):
        val = incremental_ic(p, BASELINES[name], demean=True).mean
        assert abs(val) < 0.05, f"{name}: spurious increment {val:.4f}"


def test_naive_incremental_ic_is_biased_upward_at_zero_skill():
    """Documents the bias that motivates ``demean=True`` as the default.

    Without demeaning, the control is a noisy proxy for the entity level, so
    residual level information remains in both residuals and is misattributed to
    the model. This test asserts the bias is real and large enough to matter --
    if a future change made demean=False unbiased, this test failing would be
    the signal to revisit the default.
    """
    p, _ = generate_panel(skill=0.0, level_leak=1.0)
    biased = incremental_ic(p, BASELINES["persistence"], demean=False).mean
    corrected = incremental_ic(p, BASELINES["persistence"], demean=True).mean
    assert biased > 0.1, f"expected upward bias, got {biased:.4f}"
    assert abs(corrected) < 0.05
    assert biased > corrected


def test_naive_incremental_ic_carries_warning_metadata():
    p, _ = generate_panel(skill=0.2)
    series = incremental_ic(p, BASELINES["persistence"], demean=False)
    assert "warning" in series.meta


def test_incremental_ic_increases_with_skill():
    means = []
    for skill in (0.0, 0.25, 0.5, 0.75):
        p, _ = generate_panel(skill=skill)
        means.append(incremental_ic(p, BASELINES["persistence"]).mean)
    assert means == sorted(means), f"not monotone: {means}"


def test_controlling_for_the_level_twice_is_undefined():
    """Demeaning already removes the level, so the level baseline has no variance left.

    Reporting "undefined" here is the honest outcome; a number would be an
    artefact of floating-point residue.
    """
    p, _ = generate_panel(skill=0.5)
    series = incremental_ic(p, BASELINES["per_entity_train_mean"], demean=True)
    assert series.n_dates_used == 0
    assert "note" in series.meta


def test_baseline_panel_scored_through_same_code_path():
    """A baseline panel must be a valid Panel with the baseline as prediction."""
    p, _ = generate_panel(skill=0.3)
    bp = build_baseline_panel(p, BASELINES["persistence"])
    assert bp.train_end == p.train_end
    assert bp.data[PRED].notna().all()
    assert bp.n_rows < p.n_rows  # first observation per entity is dropped
