"""Tests for the selection-bias corrections.

The central case is a grid of candidates with **no** edge at all. The best of
them will look respectable — that is what maxima of noise do — and the correction
must decline to endorse it. The paired case matters as much: the same returns,
reported as a single test rather than the winner of a search, should pass, since
there was no selection to correct for.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from audit import demo
from audit.audits.selection import (
    benjamini_hochberg,
    deflated_sharpe,
    expected_max_sharpe,
    screen_candidates,
    sharpe_estimator_variance,
)
from audit.report.text import format_selection


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
    """The correction must not condemn everything, or it would be useless.

    Old behaviour: a strong result over 42 declared trials returned
    ``passed is True``. Target behaviour: a PASS asserts that the Deflated
    Sharpe Ratio was computed, and without the Sharpe of every candidate the
    benchmark used a substituted variance, so the ceiling is INCONCLUSIVE. The
    ledger makes the same result pass, which is what this now checks: the point
    of the test is that the correction does not condemn a genuinely strong
    result, and that survives.
    Migration impact: callers reading ``passed is True`` from a ledger-free call
    now read ``None``. That is the intended narrowing -- the old True asserted a
    statistic that had not been computed -- and the remedy is to supply
    ``trial_sharpes`` rather than to relax the assertion.
    """
    rng = np.random.default_rng(1)
    strong = pd.Series(rng.normal(0.004, 0.01, 756))  # daily Sharpe ~0.4
    ledger = [float(strong.mean() / strong.std(ddof=1))] + list(
        rng.normal(0.0, 0.04, 41)
    )

    res = deflated_sharpe(strong, n_trials=42, trial_sharpes=ledger)
    assert res.passed is True
    assert res.observed_sharpe > res.expected_max_sharpe

    capped = deflated_sharpe(strong, n_trials=42)
    assert capped.passed is None
    assert capped.inconclusive_reason == "proxy_benchmark"


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


def test_screen_names_and_limits_its_iid_normal_approximation():
    table = screen_candidates(_noise_candidates())
    assert "iid_normal_pvalue" in table.columns
    assert "pvalue" not in table.columns
    assert set(table["pvalue_method"]) == {"iid_normal_approximation"}
    assert not table["strong_pass_eligible"].any()


# ----------------------------------------------------------------------
# The benchmark's variance: formal against substituted
# ----------------------------------------------------------------------
def _strong_candidate_set(seed: int = 3):
    """A winner good enough to clear its benchmark, plus the ledger behind it."""
    rng = np.random.default_rng(seed)
    grid = {f"cfg{i:02d}": pd.Series(rng.normal(0.0, 0.01, 2000)) for i in range(11)}
    grid["cfg_real"] = pd.Series(rng.normal(0.0035, 0.01, 2000))
    sharpes = [float(v.mean() / v.std(ddof=1)) for v in grid.values()]
    winner = max(grid, key=lambda k: grid[k].mean() / grid[k].std(ddof=1))
    return grid[winner], sharpes


def test_a_trial_ledger_gives_the_formal_statistic_and_can_reach_pass():
    """Detection half: with V[{SR_n}] the module computes what it names."""
    returns, sharpes = _strong_candidate_set()
    res = deflated_sharpe(returns, n_trials=len(sharpes), trial_sharpes=sharpes)

    assert res.provenance.method == "formal"
    assert res.provenance.trial_variance_source == "cross_trial"
    assert res.provenance.n_trials_kind == "inferred_from_ledger"
    assert res.provenance.strong_pass_eligible is True
    assert res.passed is True, "a formal PASS must be reachable at all"
    assert "selection_adjusted_sharpe_per_period" in res.to_dict()
    assert "approximate_selection_adjusted_sharpe_per_period" not in res.to_dict()


def test_without_a_ledger_the_same_returns_are_capped_at_inconclusive():
    """Control half: identical data, substituted benchmark, no strong verdict.

    The only difference between this and the case above is whether the Sharpe of
    every candidate was supplied. The arithmetic is unchanged; what changes is
    which quantity the benchmark estimates, and therefore what may be claimed.
    """
    returns, sharpes = _strong_candidate_set()
    res = deflated_sharpe(returns, n_trials=len(sharpes))

    assert res.provenance.method == "proxy"
    assert res.provenance.trial_variance_source == "winner_estimator"
    assert res.provenance.n_trials_kind == "declared"
    assert res.provenance.strong_pass_eligible is False

    assert res.passed is None, "PASS is unreachable without the cross-trial variance"
    assert res.inconclusive_reason == "proxy_benchmark"
    assert res.deflated_probability > 0.95, "it is capped, not failing on the evidence"

    # Renamed, because the figure is not the formally defined one.
    assert "approximate_selection_adjusted_sharpe_per_period" in res.to_dict()
    assert "selection_adjusted_sharpe_per_period" not in res.to_dict()

    rendered = format_selection(res)
    # The marker, not the word: the verdict text legitimately mentions PASS
    # when explaining what would make one reachable.
    assert "[PASS]" not in rendered
    assert "[----]" in rendered
    assert "INCONCLUSIVE" in rendered
    assert "SKIPPED" not in rendered


def test_the_two_inconclusive_causes_stay_distinguishable():
    """Criterion 4 one level down: one word, two remedies, kept apart.

    "Could not be computed" is answered with more observations; "computed from a
    proxy benchmark" is answered with a trial ledger. Collapsing them would tell
    a reader to do the wrong thing.
    """
    returns, sharpes = _strong_candidate_set()
    capped = deflated_sharpe(returns, n_trials=len(sharpes))
    uncomputable = deflated_sharpe(pd.Series([0.01, -0.01, 0.02]), n_trials=10)

    assert capped.passed is uncomputable.passed is None
    assert capped.inconclusive_reason == "proxy_benchmark"
    assert uncomputable.inconclusive_reason == "not_computable"


def test_a_single_trial_keeps_a_strong_verdict_without_a_ledger():
    """The one honest exemption, and why it is not a loophole.

    ``expected_max_sharpe`` returns 0.0 for one trial whatever variance it is
    given, so the formal and substituted benchmarks are identical rather than
    approximately equal. Capping here would fail an honest single test for a
    substitution that could not have affected it -- the false-alarm failure of
    incidents 4 and 5.
    """
    returns, _ = _strong_candidate_set()
    res = deflated_sharpe(returns, n_trials=1)

    assert res.provenance.method == "proxy"
    assert res.provenance.strong_pass_eligible is True
    assert res.expected_max_sharpe == 0.0
    assert res.passed is True


def test_a_ledger_that_disagrees_with_n_trials_is_an_error():
    """The disagreement the ledger exists to prevent must not pass silently."""
    returns, sharpes = _strong_candidate_set()
    with pytest.raises(ValueError, match="ledger is the authority"):
        deflated_sharpe(returns, n_trials=len(sharpes) + 1, trial_sharpes=sharpes)


def test_the_cross_trial_variance_is_what_the_benchmark_uses():
    """Ties the field name to the arithmetic, not just to a label.

    The two variances differ by orders of magnitude in both directions -- a
    correlated grid shrinks the cross-trial variance, a heterogeneous set
    enlarges it -- so a benchmark built from the wrong one is not cosmetically
    wrong. This pins that the ledger path actually uses V[{SR_n}].
    """
    returns, _ = _strong_candidate_set()
    spread = [0.0, 0.5, -0.4, 0.9, -0.8, 0.3, 0.1, -0.2, 0.6, -0.5, 0.2, 0.4]
    res = deflated_sharpe(returns, n_trials=len(spread), trial_sharpes=spread)

    expected = expected_max_sharpe(len(spread), float(np.var(spread, ddof=1)))
    assert res.expected_max_sharpe == pytest.approx(expected)


def test_every_provenance_field_reaches_both_rendered_surfaces():
    """Generic over the fields, so a fifth surface needs no edit here.

    Four renderings of one fact -- dataclass, text, JSON, coverage verdict -- is
    the shape that drifted three times in the previous commit. The check is
    written against ``provenance.to_dict()`` rather than against the four field
    names, so adding a field cannot leave a renderer behind.
    """
    returns, sharpes = _strong_candidate_set()
    for res in (
        deflated_sharpe(returns, n_trials=len(sharpes)),
        deflated_sharpe(returns, n_trials=len(sharpes), trial_sharpes=sharpes),
    ):
        serialised = res.to_dict()
        rendered = format_selection(res)
        for key, value in res.provenance.to_dict().items():
            assert serialised[key] == value, f"{key} missing from the JSON report"
            assert key in rendered, f"{key} missing from the text report"
        for label, shown in res.provenance.as_rows():
            assert shown in rendered, f"{label} value {shown!r} missing from the text"


def test_the_demo_shows_the_cap_downgrading_a_verdict(tmp_path):
    """Criterion 5: behaviour visible only in tests is not delivered.

    The 42-noise case fails on its own evidence, so it shows the cap's fields
    without ever showing the cap doing anything. A reader of the demo output
    could not tell the ceiling had any effect. This pins the pair that shows it:
    the same winner, scored with and without the ledger, both clearing the PASS
    threshold, one downgraded.
    """
    demo.main(["--outdir", str(tmp_path)])

    capped = json.loads(
        (tmp_path / "selection_capped_report.json").read_text(encoding="utf-8")
    )
    formal = json.loads(
        (tmp_path / "selection_ledger_report.json").read_text(encoding="utf-8")
    )

    # Same data, so the evidence clears the threshold in both.
    assert capped["deflated_probability"] > 0.95
    assert formal["deflated_probability"] > 0.95

    assert capped["method"] == "proxy"
    assert capped["passed"] is None
    assert capped["inconclusive_reason"] == "proxy_benchmark"
    assert capped["strong_pass_eligible"] is False

    assert formal["method"] == "formal"
    assert formal["passed"] is True
    assert formal["strong_pass_eligible"] is True

    text = (tmp_path / "selection_capped_report.txt").read_text(encoding="utf-8")
    assert "[PASS]" not in text
    assert "SKIPPED" not in text
