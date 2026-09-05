"""Selection bias: is the best of N candidates better than the best of N coin flips?

Every module before this one audits a single result. This one audits the *search*
that produced it.

Scan a 7x6 parameter grid, report the best cell, and the number you report is not
an estimate of that configuration's performance — it is the maximum of 42 noisy
estimates. Maxima of noise are large. With 42 independent candidates that have no
edge whatsoever, the best one will show a Sharpe around 2 on a few years of daily
data, purely from the dispersion of the sampling distribution.

Nothing about the winning backtest looks wrong. The equity curve is real, the
trades are real, the Sharpe is arithmetically correct. What is wrong is the
inference: the configuration was chosen *because* it scored highest, so its score
carries the selection.

Two corrections, for two situations
-----------------------------------
**Deflated Sharpe Ratio** — for "I tried N configurations and am reporting the
best." It asks what the maximum Sharpe would be under the null of no skill, given
N trials and the observed non-normality of the returns, and expresses the
observed Sharpe as a probability of exceeding that benchmark. Following Bailey
and López de Prado.

**Benjamini-Hochberg** — for "I tested N candidate signals and want to know which
survive." Controlling the family-wise error rate (Bonferroni) is too strict when
N is large and the goal is a shortlist rather than a single verdict; controlling
the false discovery rate answers the question actually being asked: of the
signals I keep, what share are expected to be spurious?

Where distributional assumptions differ
-----------------------------------------
The expected maximum under the null depends on the shape of the return
distribution, not only on N. Strategy returns are typically negatively skewed
with fat tails — which inflates the variance of the Sharpe estimator and
therefore raises the bar the observed Sharpe has to clear. Ignoring skew and
kurtosis makes the correction too lenient in exactly the cases where a correction
matters most, so the Deflated Sharpe path estimates both from the returns.

The ``benjamini_hochberg`` function is assumption-agnostic: callers may supply
HAC or bootstrap p-values. The convenience ``screen_candidates`` function does
not estimate those robust p-values; it uses a one-sided iid-normal approximation
and labels that field explicitly. Its shortlist is exploratory and is never, by
itself, evidence for a strong PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..metrics.performance import annualise_sharpe

# Euler-Mascheroni constant, used in the expected-maximum approximation.
EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class SelectionProvenance:
    """How the benchmark was computed, and what verdict that permits.

    One source for four renderings. These fields appear in the dataclass, the
    text report, the JSON report and the coverage verdict, and every one of them
    projects from ``to_dict`` rather than restating the values or re-deriving
    the rule that connects them. Four renderings of one fact is the shape that
    has already drifted three times in this repository.
    """

    method: str  # "formal" | "proxy"
    trial_variance_source: str  # "cross_trial" | "winner_estimator"
    n_trials_kind: str  # "declared" | "inferred_from_ledger"
    strong_pass_eligible: bool

    @classmethod
    def formal(cls) -> SelectionProvenance:
        return cls("formal", "cross_trial", "inferred_from_ledger", True)

    @classmethod
    def proxy(cls, *, benchmark_is_variance_free: bool) -> SelectionProvenance:
        """The substituted benchmark.

        ``benchmark_is_variance_free`` is the one case where a strong verdict
        survives without a ledger: with a single trial ``expected_max_sharpe``
        returns 0.0 whatever variance it is given, so the formal and substituted
        benchmarks are not approximately equal but identical, and there is no
        selection to deflate. Capping it would fail an honest single test for a
        substitution that cannot have affected it.
        """
        return cls("proxy", "winner_estimator", "declared", benchmark_is_variance_free)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "trial_variance_source": self.trial_variance_source,
            "n_trials_kind": self.n_trials_kind,
            "strong_pass_eligible": self.strong_pass_eligible,
        }

    def as_rows(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for any renderer, derived from ``to_dict``."""
        return tuple(
            (key, str(value).lower() if isinstance(value, bool) else str(value))
            for key, value in self.to_dict().items()
        )


@dataclass
class DeflatedSharpeResult:
    """Observed Sharpe against the maximum expected from selection alone.

    Every Sharpe here is **per period**. The deflation compares an observed
    Sharpe against the maximum expected from selection over ``n_observations``,
    so both sides must be on the per-period scale; annualising either would
    break the comparison. The annualised figure is carried separately, and only
    when a caller supplied the observation frequency, because a report that
    prints one number under a bare "Sharpe" heading leaves a reader unable to
    tell which of the two by a factor of sqrt(periods_per_year) it is.
    """

    observed_sharpe: float
    n_trials: int
    n_observations: float
    expected_max_sharpe: float
    selection_adjusted_sharpe: float
    deflated_probability: float
    skew: float
    kurtosis: float
    passed: bool | None
    verdict: str
    periods_per_year: int | None = None
    observed_sharpe_annualised: float | None = None
    provenance: SelectionProvenance = field(default_factory=SelectionProvenance.formal)
    # INCONCLUSIVE has two causes with different remedies: the statistic could
    # not be computed, or it was computed from a substituted benchmark. Keeping
    # the reason machine-readable is what stops one word covering both.
    inconclusive_reason: str | None = None
    detail: dict = field(default_factory=dict)

    @property
    def selection_adjusted_sharpe_key(self) -> str:
        """The headline figure is not called by a formal name under a proxy.

        The ``_per_period`` suffix is not decoration. Every Sharpe this
        repository reports declares its scale, and the surface-scanning test
        rejected this key without it -- correctly, since a reader of the JSON
        would otherwise have to infer whether an adjusted Sharpe was per period
        or annualised. The distinction the name carries is
        ``approximate_`` against formal, and that is untouched by the suffix.
        """
        if self.provenance.method == "formal":
            return "selection_adjusted_sharpe_per_period"
        return "approximate_selection_adjusted_sharpe_per_period"

    def to_dict(self) -> dict:
        return {
            self.selection_adjusted_sharpe_key: self.selection_adjusted_sharpe,
            **self.provenance.to_dict(),
            "inconclusive_reason": self.inconclusive_reason,
            "observed_sharpe_per_period": self.observed_sharpe,
            "observed_sharpe_annualised": self.observed_sharpe_annualised,
            "periods_per_year": self.periods_per_year,
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "expected_max_sharpe_per_period": self.expected_max_sharpe,
            "deflated_probability": self.deflated_probability,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "passed": self.passed,
            "verdict": self.verdict,
            **self.detail,
        }


def expected_max_sharpe(n_trials: int, variance_of_sharpe: float = 1.0) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent null strategies.

    ``variance_of_sharpe`` is, in Bailey and Lopez de Prado, ``V[{SR_n}]``: the
    variance *across the N trials'* estimated Sharpe ratios. It is a property of
    the search, not of any one candidate, and computing it requires the Sharpe
    of every configuration examined.

    Callers without a trial ledger substitute the sampling variance of the
    winner's own Sharpe estimator. The two coincide when the trials are
    independent and identically distributed under the null, and not otherwise.
    A correlated parameter grid makes the cross-trial variance smaller, which
    makes the substituted benchmark too high; a heterogeneous candidate set
    makes it larger, which makes the substituted benchmark too low. The
    direction of the resulting error is therefore not determined a priori, and
    this function cannot tell which it was handed.

    Uses the standard extreme-value approximation for the maximum of N standard
    normals:

        E[max] ≈ (1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))

    scaled by the standard deviation of the Sharpe estimator. The approximation
    is good for N of a few and excellent for N in the hundreds, which spans the
    range in which parameter grids are actually scanned.

    Returns 0.0 for a single trial: with no selection there is nothing to deflate,
    and returning a positive benchmark would penalise an honest single test.
    """
    if n_trials <= 1:
        return 0.0
    sd = np.sqrt(max(variance_of_sharpe, 0.0))
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def sharpe_estimator_variance(n_obs: int, sharpe: float, skew: float, kurtosis: float) -> float:
    """Variance of the Sharpe estimator under non-normal returns.

        Var[SR] ≈ (1 - γ₃·SR + (γ₄-1)/4·SR²) / (n - 1)

    where γ₃ is skewness and γ₄ is kurtosis (not excess). Negative skew and fat
    tails both raise it, which is why a correction that assumed normality would be
    too lenient on precisely the return distributions that need it most.
    """
    if n_obs < 2:
        return float("nan")
    return float(
        (1.0 - skew * sharpe + 0.25 * (kurtosis - 1.0) * sharpe**2) / (n_obs - 1)
    )


def deflated_sharpe(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    threshold_sharpe: float = 0.0,
    periods_per_year: int | None = None,
    trial_sharpes: pd.Series | np.ndarray | None = None,
) -> DeflatedSharpeResult:
    """Probability the observed Sharpe exceeds what selection alone would produce.

    ``n_trials`` is the number of configurations actually examined — the honest
    count, including the ones abandoned early. Understating it is the easiest way
    to make this correction say what one wants, and there is no way to detect that
    from the returns.

    The benchmark, and what is substituted for it
    ---------------------------------------------
    Bailey and Lopez de Prado define the benchmark ``SR*`` in terms of
    ``V[{SR_n}]``, the variance of the estimated Sharpe ratios *across the N
    trials*. That is a property of the search: it needs the Sharpe of every
    candidate examined, which needs a trial ledger. This repository has
    deliberately deferred that ledger, so ``PLAN.md`` records it as the
    precondition for computing ``SR*`` as defined rather than as a matter of
    traceability.

    Supply ``trial_sharpes`` and the result is the formal Deflated Sharpe Ratio.
    Omit it and the sampling variance of the *winner's own* Sharpe estimator is
    substituted. The two coincide when the trials are independent and
    identically distributed under the null, and not otherwise: a correlated grid
    shrinks the cross-trial variance and so raises the substituted benchmark
    above the formal one, while a heterogeneous candidate set enlarges it and so
    lowers the substituted benchmark below it. The direction of the error is not
    determined a priori and cannot be recovered from the winning return stream.

    So the substitution is kept, and the claim is not. Without a ledger the
    result is marked ``method=proxy``, the headline figure is renamed
    ``approximate_selection_adjusted_sharpe``, and the verdict is capped at
    INCONCLUSIVE — a strong PASS would assert the formal quantity had been
    computed when it had not. This is the same candour as the
    ``iid_normal_pvalue`` field in ``screen_candidates``: the approximation is
    usable and is never, by itself, evidence for a strong PASS.

    FAIL remains reachable, deliberately and asymmetrically. A substituted
    benchmark can be too high as easily as too low, so a FAIL is not formally
    warranted either; it is retained as a caution rather than a certification,
    because the failure this framework exists to prevent is endorsing a result,
    not doubting one. The verdict text says so.

    The z-statistic below is unaffected: its denominator is the sampling
    variance of the estimator, which is what BLdP specify there.

    ``periods_per_year`` does not enter the computation, which must stay on the
    per-period scale. It only lets the report state the annualised figure a
    reader is otherwise left to infer.
    """
    if trial_sharpes is not None:
        ledger = np.asarray(pd.Series(trial_sharpes).dropna(), dtype=float)
        if ledger.size != n_trials:
            raise ValueError(
                f"trial_sharpes has {ledger.size} entries but n_trials is "
                f"{n_trials}; the ledger is the authority on how many "
                "configurations were examined, and a silent disagreement here "
                "is the error the ledger exists to prevent"
            )
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = x.size

    if n < 10:
        return DeflatedSharpeResult(
            observed_sharpe=float("nan"),
            selection_adjusted_sharpe=float("nan"),
            provenance=(
                SelectionProvenance.formal()
                if trial_sharpes is not None
                else SelectionProvenance.proxy(benchmark_is_variance_free=n_trials <= 1)
            ),
            inconclusive_reason="not_computable",
            n_trials=n_trials,
            n_observations=n,
            expected_max_sharpe=float("nan"),
            deflated_probability=float("nan"),
            skew=float("nan"),
            kurtosis=float("nan"),
            passed=None,
            verdict=f"INCONCLUSIVE: {n} observations is too few to estimate a Sharpe.",
            periods_per_year=periods_per_year,
        )

    mean, sd = float(x.mean()), float(x.std(ddof=1))
    observed = mean / sd if sd > 0 else float("nan")
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x, fisher=False))

    # Sampling variance of this estimator: BLdP's denominator for the
    # z-statistic, and correct there whether or not a ledger was supplied.
    var_sr = sharpe_estimator_variance(n, observed, skew, kurt)

    # The benchmark's variance is a different quantity: V[{SR_n}] across trials.
    # With a ledger it is computed; without one the estimator variance stands in
    # and the result says so rather than claiming the formal statistic.
    if trial_sharpes is not None:
        benchmark_variance = float(np.var(ledger, ddof=1)) if ledger.size > 1 else 0.0
        provenance = SelectionProvenance.formal()
    else:
        benchmark_variance = var_sr
        # At one trial `expected_max_sharpe` returns 0.0 for any variance, so
        # the substituted and formal benchmarks are identical rather than close.
        provenance = SelectionProvenance.proxy(benchmark_is_variance_free=n_trials <= 1)
    benchmark = expected_max_sharpe(n_trials, benchmark_variance)
    adjusted = observed - max(threshold_sharpe, benchmark)

    # Probability that the observed Sharpe exceeds the selection benchmark,
    # accounting for the estimator's own uncertainty.
    if np.isfinite(var_sr) and var_sr > 0:
        z = (observed - max(threshold_sharpe, benchmark)) / np.sqrt(var_sr)
        prob = float(stats.norm.cdf(z))
    else:
        prob = float("nan")

    inconclusive_reason = None
    if not np.isfinite(prob):
        passed = None
        inconclusive_reason = "not_computable"
        verdict = "INCONCLUSIVE: the deflated probability could not be computed."
    elif prob > 0.95 and not provenance.strong_pass_eligible:
        # Computed, and insufficient -- not uncomputed. The remedy is a ledger.
        passed = None
        inconclusive_reason = "proxy_benchmark"
        verdict = (
            f"INCONCLUSIVE: Sharpe {observed:.2f} per period over {n_trials} "
            f"trials clears the {benchmark:.2f} benchmark with deflated "
            f"probability {prob:.3f}, but that benchmark used the winner's own "
            "estimator variance in place of the variance across the trials. "
            "The Deflated Sharpe Ratio is defined on the second, so this is an "
            "approximation and not the formal statistic. Supply the Sharpe of "
            "every configuration examined and a PASS becomes reachable."
        )
    elif prob > 0.95:
        passed = True
        verdict = (
            f"PASS: Sharpe {observed:.2f} per period over {n_trials} trials, against "
            f"an expected maximum of {benchmark:.2f} from selection alone. "
            f"Deflated probability {prob:.3f} -- the result survives the "
            "correction for having chosen the best of several candidates."
        )
    elif prob > 0.5:
        passed = None
        inconclusive_reason = "below_threshold"
        verdict = (
            f"INCONCLUSIVE: Sharpe {observed:.2f} per period against a selection "
            f"benchmark of {benchmark:.2f}, deflated probability {prob:.3f}. Above "
            "the benchmark but not decisively; more out-of-sample data is the only "
            "thing that settles this."
        )
    else:
        passed = False
        verdict = (
            f"FAIL: Sharpe {observed:.2f} per period does not clear the "
            f"{benchmark:.2f} expected from picking the best of {n_trials} trials "
            f"(deflated probability {prob:.3f}). The reported figure is "
            "consistent with having selected the luckiest configuration rather "
            "than a skilful one."
        )

    if passed is False and provenance.method == "proxy":
        verdict += (
            " The benchmark is an approximation: it used the winner's estimator "
            "variance rather than the variance across trials, and the direction "
            "of that error is not determined. Read this as a caution, not as a "
            "formal rejection."
        )

    return DeflatedSharpeResult(
        observed_sharpe=observed,
        selection_adjusted_sharpe=adjusted,
        provenance=provenance,
        inconclusive_reason=inconclusive_reason,
        n_trials=n_trials,
        n_observations=n,
        expected_max_sharpe=benchmark,
        deflated_probability=prob,
        skew=skew,
        kurtosis=kurt,
        passed=passed,
        verdict=verdict,
        periods_per_year=periods_per_year,
        observed_sharpe_annualised=annualise_sharpe(observed, periods_per_year),
        detail={
            "sharpe_per_period_estimator_variance": var_sr,
            "threshold_sharpe_per_period": threshold_sharpe,
        },
    )


def benjamini_hochberg(pvalues: dict[str, float], fdr: float = 0.10) -> pd.DataFrame:
    """Which candidates survive at a given false discovery rate.

    Controls the expected share of false positives *among the rejections*, which
    is the quantity of interest when the output is a shortlist. Bonferroni
    controls the probability of any false positive at all — far stricter, and the
    wrong target when screening many candidates, since it discards genuine
    signals to buy a guarantee nobody asked for.

    Returns a frame sorted by p-value with the BH threshold and a survival flag.
    """
    if not pvalues:
        return pd.DataFrame(columns=["pvalue", "rank", "bh_threshold", "survives"])

    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    rows = []
    for i, (name, p) in enumerate(items, start=1):
        rows.append(
            {"candidate": name, "pvalue": p, "rank": i, "bh_threshold": fdr * i / n}
        )
    table = pd.DataFrame(rows).set_index("candidate")

    # The BH step-up: find the largest rank whose p-value clears its threshold,
    # then reject everything up to it. Testing each row independently would be
    # wrong -- the procedure is defined by the largest passing rank, not by
    # per-row comparison.
    passing = table.loc[table["pvalue"] <= table["bh_threshold"], "rank"]
    cutoff = int(passing.max()) if not passing.empty else 0
    table["survives"] = table["rank"] <= cutoff

    return table


def screen_candidates(
    returns_by_candidate: dict[str, pd.Series],
    fdr: float = 0.10,
) -> pd.DataFrame:
    """Exploratory screen using one-sided iid-normal approximate p-values.

    The gap between the two columns is the point: the count of candidates that
    look significant individually, against the count that survive once the size
    of the search is taken into account. Serial dependence and non-normality are
    not corrected here; callers with HAC or bootstrap p-values should pass them
    directly to ``benjamini_hochberg``.
    """
    pvalues, sharpes = {}, {}
    for name, series in returns_by_candidate.items():
        x = np.asarray(pd.Series(series).dropna(), dtype=float)
        if x.size < 10 or x.std(ddof=1) <= 0:
            continue
        sr = x.mean() / x.std(ddof=1)
        tstat = sr * np.sqrt(x.size)
        pvalues[name] = float(1.0 - stats.norm.cdf(tstat))  # one-sided
        sharpes[name] = float(sr)

    table = benjamini_hochberg(pvalues, fdr=fdr).rename(
        columns={"pvalue": "iid_normal_pvalue"}
    )
    table["sharpe_per_period"] = pd.Series(sharpes)
    table["naive_significant"] = table["iid_normal_pvalue"] < 0.05
    table["pvalue_method"] = "iid_normal_approximation"
    table["strong_pass_eligible"] = False
    return table[
        [
            "sharpe_per_period",
            "iid_normal_pvalue",
            "rank",
            "bh_threshold",
            "naive_significant",
            "survives",
            "pvalue_method",
            "strong_pass_eligible",
        ]
    ]
