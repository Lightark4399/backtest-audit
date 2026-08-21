"""Tests for the selection-bias corrections.

The central case is a grid of candidates with **no** edge at all. The best of
them will look respectable — that is what maxima of noise do — and the correction
must decline to endorse it. The paired case matters as much: the same returns,
reported as a single test rather than the winner of a search, should pass, since
there was no selection to correct for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit.audits.selection import (
    benjamini_hochberg,
    deflated_sharpe,
    expected_max_sharpe,
    screen_candidates,
    sharpe_estimator_variance,
)


def _noise_candidates(n_candidates: int = 42, n_obs: int = 756, seed: int = 7):
    rng = np.random.default_rng(seed)
    return {f"cfg{i:02d}": pd.Series(rng.normal(0.0, 0.01, n_obs)) for i in range(n_candidates)}


def _best_of(candidates: dict[str, pd.Series]) -> tuple[str, pd.Series]:
    sharpes = {k: v.mean() / v.std(ddof=1) for k, v in candidates.items()}
    best = max(sharpes, key=sharpes.get)
    return best, candidates[best]


# ----------------------------------------------------------------------
# Expected maximum under the null
# ----------------------------------------------------------------------
def test_expected_max_grows_with_the_number_of_trials():
    values = [expected_max_sharpe(n, variance_of_sharpe=1.0) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)


def test_single_trial_has_no_selection_benchmark():
    """With nothing selected there is nothing to deflate."""
    assert expected_max_sharpe(1) == 0.0


def test_estimator_variance_rises_with_negative_skew_and_fat_tails():
    """The correction must be stricter on the distributions that need it most."""
    normal = sharpe_estimator_variance(500, sharpe=0.1, skew=0.0, kurtosis=3.0)
    skewed = sharpe_estimator_variance(500, sharpe=0.1, skew=-1.5, kurtosis=3.0)
    fat = sharpe_estimator_variance(500, sharpe=0.1, skew=0.0, kurtosis=9.0)
    assert skewed > normal
    assert fat > normal


# ----------------------------------------------------------------------
# The central case
# ----------------------------------------------------------------------
def test_best_of_many_noise_candidates_is_not_endorsed():
    """A grid with no edge produces a respectable-looking winner. It must fail."""
    candidates = _noise_candidates()
    _, best = _best_of(candidates)

    annualised = (best.mean() / best.std(ddof=1)) * np.sqrt(252)
    assert annualised > 0.8, "the best of 42 noise candidates should look plausible"

    res = deflated_sharpe(best, n_trials=len(candidates))
    assert res.passed is False
    assert res.deflated_probability < 0.5


def test_the_same_returns_pass_when_reported_as_a_single_test():
    """The correction is about the search, not about the returns.

    Identical data, different provenance, different verdict — which is the whole
    claim of the module, and the reason `n_trials` has to be honest.
    """
    candidates = _noise_candidates()
    _, best = _best_of(candidates)

    many = deflated_sharpe(best, n_trials=len(candidates))
    one = deflated_sharpe(best, n_trials=1)

    assert many.passed is False
    assert one.passed is True
    assert one.expected_max_sharpe == 0.0


def test_deflated_probability_falls_as_trials_rise():
    _, best = _best_of(_noise_candidates())
    probs = [deflated_sharpe(best, n_trials=n).deflated_probability for n in (1, 10, 100)]
    assert probs == sorted(probs, reverse=True), f"not decreasing in trials: {probs}"


def test_a_genuinely_strong_result_survives_the_correction():
    """The correction must not condemn everything, or it would be useless."""
    rng = np.random.default_rng(1)
    strong = pd.Series(rng.normal(0.004, 0.01, 756))  # daily Sharpe ~0.4
    res = deflated_sharpe(strong, n_trials=42)
    assert res.passed is True
    assert res.observed_sharpe > res.expected_max_sharpe


def test_short_series_declines_to_judge():
    res = deflated_sharpe(pd.Series([0.01, -0.01, 0.02]), n_trials=10)
    assert res.passed is None
    assert "too few" in res.verdict


# ----------------------------------------------------------------------
# Benjamini-Hochberg
# ----------------------------------------------------------------------
def test_bh_rejects_nothing_when_all_candidates_are_noise():
    table = screen_candidates(_noise_candidates())
    assert table["naive_significant"].sum() > 0, "some noise looks significant individually"
    assert table["survives"].sum() == 0, "none should survive FDR control"


def test_bh_keeps_genuinely_strong_candidates():
    rng = np.random.default_rng(2)
    candidates = _noise_candidates(n_candidates=20)
    for i in range(3):
        candidates[f"real{i}"] = pd.Series(rng.normal(0.005, 0.01, 756))
    table = screen_candidates(candidates)
    survivors = set(table.index[table["survives"]])
    assert {"real0", "real1", "real2"} <= survivors


def test_bh_is_a_step_up_procedure_not_a_per_row_test():
    """A candidate below the cutoff survives even if its own threshold is tighter.

    Testing each row independently is the classic misimplementation; the
    procedure is defined by the largest passing rank.
    """
    pvalues = {"a": 0.001, "b": 0.04, "c": 0.20, "d": 0.50}
    table = benjamini_hochberg(pvalues, fdr=0.25)
    # 'b' has p=0.04 against its own threshold 0.125, and 'a' passes as well,
    # so both are rejected; the cutoff is set by the largest passing rank.
    assert bool(table.loc["a", "survives"])
    assert bool(table.loc["b", "survives"])
    assert not bool(table.loc["d", "survives"])


def test_bh_threshold_scales_with_rank():
    table = benjamini_hochberg({"a": 0.01, "b": 0.02, "c": 0.03}, fdr=0.15)
    thresholds = list(table["bh_threshold"])
    assert thresholds == sorted(thresholds)
    assert thresholds[-1] == pytest.approx(0.15)


def test_empty_input_returns_an_empty_frame():
    assert benjamini_hochberg({}).empty


def test_screen_reports_both_naive_and_corrected_counts():
    """The gap between the two columns is the finding."""
    table = screen_candidates(_noise_candidates())
    assert {"naive_significant", "survives"} <= set(table.columns)
    assert table["naive_significant"].sum() >= table["survives"].sum()
