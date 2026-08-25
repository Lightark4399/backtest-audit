"""Semantic tests for translating audit numbers into human conclusions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit.metrics.ic import ICSeries
from audit.report.text import format_interpretation


def _series(value: float | None, name: str = "ic") -> ICSeries:
    if value is None:
        values = pd.Series(dtype=float)
        n_obs = pd.Series(dtype=int)
        undefined = 1
    else:
        values = pd.Series([value], index=[pd.Timestamp("2024-01-02")])
        n_obs = pd.Series([20], index=values.index)
        undefined = 0
    return ICSeries(name, values, n_obs, 1, undefined, 0)


def _reading(demeaned: float | None) -> str:
    baselines = pd.DataFrame({"mean": [0.10]}, index=["persistence"])
    return format_interpretation(_series(0.20, "raw"), baselines, _series(demeaned))


@pytest.mark.parametrize("value", [-0.6880, -0.02])
def test_negative_demeaned_ic_is_never_described_as_positive_skill(value):
    reading = _reading(value).lower()
    assert "negative association" in reading
    assert "small but non-zero skill" not in reading
    assert "separate out-of-sample validation" in reading


@pytest.mark.parametrize("value", [-0.0199, 0.0, 0.0199])
def test_near_zero_demeaned_ic_is_described_as_no_information(value):
    reading = _reading(value).lower()
    assert "indistinguishable from zero" in reading
    assert "negative association" not in reading


def test_undefined_demeaned_ic_declines_to_interpret():
    reading = _reading(None).lower()
    assert "could not be computed" in reading
    assert "contains real information" not in reading


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.02, "small positive association"), (0.0999, "small positive association"),
     (0.10, "substantial positive association")],
)
def test_positive_interpretation_respects_declared_thresholds(value, expected):
    assert expected in _reading(value).lower()


def test_non_finite_mean_declines_to_interpret():
    reading = _reading(np.nan).lower()
    assert "could not be computed" in reading


def test_negative_best_baseline_is_not_described_as_free_positive_score():
    baselines = pd.DataFrame({"mean": [-0.0086, -0.0090]}, index=["persistence", "ewma"])
    reading = format_interpretation(_series(0.48, "raw"), baselines, _series(0.46)).lower()
    assert "no naive baseline achieved a positive ic" in reading
    assert "-2% of the headline" not in reading
