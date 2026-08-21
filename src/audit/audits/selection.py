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

Why non-normality matters here
------------------------------
The expected maximum under the null depends on the shape of the return
distribution, not only on N. Strategy returns are typically negatively skewed
with fat tails — which inflates the variance of the Sharpe estimator and
therefore raises the bar the observed Sharpe has to clear. Ignoring skew and
kurtosis makes the correction too lenient in exactly the cases where a correction
matters most, so both are estimated from the returns rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# Euler-Mascheroni constant, used in the expected-maximum approximation.
EULER_GAMMA = 0.5772156649015329


@dataclass
class DeflatedSharpeResult:
    """Observed Sharpe against the maximum expected from selection alone."""

    observed_sharpe: float
    n_trials: int
    n_observations: float
    expected_max_sharpe: float
    deflated_probability: float
    skew: float
    kurtosis: float
    passed: bool | None
    verdict: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "observed_sharpe": self.observed_sharpe,
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "expected_max_sharpe": self.expected_max_sharpe,
            "deflated_probability": self.deflated_probability,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "passed": self.passed,
            "verdict": self.verdict,
            **self.detail,
        }


def expected_max_sharpe(n_trials: int, variance_of_sharpe: float = 1.0) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent null strategies.

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
) -> DeflatedSharpeResult:
    """Probability the observed Sharpe exceeds what selection alone would produce.

    ``n_trials`` is the number of configurations actually examined — the honest
    count, including the ones abandoned early. Understating it is the easiest way
    to make this correction say what one wants, and there is no way to detect that
    from the returns.
    """
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = x.size

    if n < 10:
        return DeflatedSharpeResult(
            observed_sharpe=float("nan"),
            n_trials=n_trials,
            n_observations=n,
            expected_max_sharpe=float("nan"),
            deflated_probability=float("nan"),
            skew=float("nan"),
            kurtosis=float("nan"),
            passed=None,
            verdict=f"INCONCLUSIVE: {n} observations is too few to estimate a Sharpe.",
        )

    mean, sd = float(x.mean()), float(x.std(ddof=1))
    observed = mean / sd if sd > 0 else float("nan")
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x, fisher=False))

    var_sr = sharpe_estimator_variance(n, observed, skew, kurt)
    benchmark = expected_max_sharpe(n_trials, var_sr)

    # Probability that the observed Sharpe exceeds the selection benchmark,
    # accounting for the estimator's own uncertainty.
    if np.isfinite(var_sr) and var_sr > 0:
        z = (observed - max(threshold_sharpe, benchmark)) / np.sqrt(var_sr)
        prob = float(stats.norm.cdf(z))
    else:
        prob = float("nan")

    if not np.isfinite(prob):
        passed = None
        verdict = "INCONCLUSIVE: the deflated probability could not be computed."
    elif prob > 0.95:
        passed = True
        verdict = (
            f"PASS: Sharpe {observed:.2f} over {n_trials} trials, against an "
            f"expected maximum of {benchmark:.2f} from selection alone. "
            f"Deflated probability {prob:.3f} -- the result survives the "
            "correction for having chosen the best of several candidates."
        )
    elif prob > 0.5:
        passed = None
        verdict = (
            f"INCONCLUSIVE: Sharpe {observed:.2f} against a selection benchmark "
            f"of {benchmark:.2f}, deflated probability {prob:.3f}. Above the "
            "benchmark but not decisively; more out-of-sample data is the only "
            "thing that settles this."
        )
    else:
        passed = False
        verdict = (
            f"FAIL: Sharpe {observed:.2f} does not clear the {benchmark:.2f} "
            f"expected from picking the best of {n_trials} trials "
            f"(deflated probability {prob:.3f}). The reported figure is "
            "consistent with having selected the luckiest configuration rather "
            "than a skilful one."
        )

    return DeflatedSharpeResult(
        observed_sharpe=observed,
        n_trials=n_trials,
        n_observations=n,
        expected_max_sharpe=benchmark,
        deflated_probability=prob,
        skew=skew,
        kurtosis=kurt,
        passed=passed,
        verdict=verdict,
        detail={
            "sharpe_estimator_variance": var_sr,
            "threshold_sharpe": threshold_sharpe,
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
    """Screen many candidate signals, reporting naive and FDR-controlled verdicts.

    The gap between the two columns is the point: the count of candidates that
    look significant individually, against the count that survive once the size
    of the search is taken into account.
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

    table = benjamini_hochberg(pvalues, fdr=fdr)
    table["sharpe"] = pd.Series(sharpes)
    table["naive_significant"] = table["pvalue"] < 0.05
    return table[["sharpe", "pvalue", "rank", "bh_threshold", "naive_significant", "survives"]]
