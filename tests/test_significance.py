"""Tests for HAC (Newey-West) inference on an IC series.

The property that matters: when the series is positively autocorrelated, the HAC
standard error must exceed the naive one. An audit tool that reported an inflated
t-statistic would be committing, in its own output, the kind of overstatement it
exists to detect.
"""

from __future__ import annotations

import numpy as np
import pytest

from audit.metrics.significance import auto_maxlags, newey_west_tstat


def _ar1(n: int, phi: float, seed: int = 0, mean: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x + mean


def test_hac_se_exceeds_naive_se_under_positive_autocorrelation():
    x = _ar1(500, phi=0.7, mean=0.5)
    res = newey_west_tstat(x)
    assert res.hac_se > res.naive_se
    assert res.se_inflation > 1.2
    assert res.hac_tstat < res.naive_tstat  # same mean, larger SE


def test_hac_and_naive_agree_when_series_is_iid():
    rng = np.random.default_rng(11)
    x = rng.normal(0.3, 1.0, size=800)
    res = newey_west_tstat(x)
    # No autocorrelation to correct for, so the two SEs should be close.
    assert res.se_inflation == pytest.approx(1.0, abs=0.25)


def test_autocorrelation_note_is_emitted():
    x = _ar1(300, phi=0.6, mean=0.4)
    res = newey_west_tstat(x)
    assert any("autocorrelation" in n for n in res.notes)


def test_zero_mean_series_is_not_significant():
    x = _ar1(400, phi=0.3, mean=0.0)
    res = newey_west_tstat(x)
    assert abs(res.hac_tstat) < 3.0
    assert res.hac_pvalue > 0.01


def test_short_series_declines_to_infer():
    """Refusing is better than emitting a t-statistic from four observations."""
    res = newey_west_tstat([0.1, 0.2, 0.15, 0.05])
    assert np.isnan(res.hac_tstat)
    assert res.notes and "observations" in res.notes[0]


def test_auto_maxlags_grows_with_sample_size():
    assert auto_maxlags(20) >= 1
    assert auto_maxlags(1000) > auto_maxlags(100)
    assert auto_maxlags(100) == 4  # floor(4 * 1^(2/9))


def test_maxlags_is_reported_for_reproducibility():
    """A HAC t-statistic without its bandwidth cannot be reproduced by a reviewer."""
    res = newey_west_tstat(_ar1(200, phi=0.5, mean=0.2))
    assert res.maxlags >= 1
    assert "maxlags" in res.to_dict()


def test_maxlags_reduced_when_too_large_for_sample():
    res = newey_west_tstat(_ar1(10, phi=0.3, mean=0.2), maxlags=50)
    assert res.maxlags < 10
    assert any("maxlags reduced" in n for n in res.notes)


def test_nan_values_are_dropped_not_propagated():
    x = np.array([0.1, np.nan, 0.2, 0.3, 0.15, np.nan, 0.25, 0.2])
    res = newey_west_tstat(x)
    assert res.n_obs == 6
    assert np.isfinite(res.mean)
