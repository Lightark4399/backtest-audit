"""Significance testing for a daily IC series.

The claim "mean IC is positive" is a claim about the mean of a time series, and
the naive t-statistic ``mean / (std / sqrt(T))`` assumes those T observations are
independent. Daily cross-sectional ICs are not independent: they inherit
autocorrelation from persistent features, from slowly-varying market regimes, and
from the fact that a model retrained infrequently makes systematically similar
errors on consecutive days.

Serial dependence changes the standard error of the mean. Whether the naive
standard error is too small or too large depends on the full dependence
structure -- the sign and size of the autocovariances at every lag -- not on the
lag-1 value alone. A heteroskedasticity- and autocorrelation-consistent (HAC /
Newey-West) standard error, obtained by regressing the IC series on a constant,
estimates it from the data up to ``maxlags``.

This matters for the framework's purpose: a t-statistic computed under the
wrong independence assumption can make a backtest look more or less trustworthy
than it is, so the tool reports both the naive and the HAC statistic side by
side and makes the difference visible.

Lag selection
-------------
The default follows the common automatic rule ``floor(4 * (T/100)^(2/9))``,
which grows slowly with sample size. It is a rule of thumb, not an optimum, so it
is configurable and always reported alongside the statistic -- a HAC t-statistic
without its lag count is not reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_OBS_FOR_TEST = 5


@dataclass
class SignificanceResult:
    """Naive and HAC inference for the mean of a series."""

    n_obs: int
    mean: float
    naive_se: float
    naive_tstat: float
    hac_se: float
    hac_tstat: float
    hac_pvalue: float
    maxlags: int
    lag1_autocorr: float
    notes: list[str] = field(default_factory=list)

    @property
    def effective_n(self) -> float:
        """AR(1) approximation to the independent-observation equivalent of the sample.

        Assuming the series is AR(1) with lag-1 autocorrelation rho,

            n_eff = n * (1 - rho) / (1 + rho)

        This is a heuristic diagnostic, not a restatement of the HAC correction.
        It sees only lag 1, whereas the Newey-West standard error weights every
        lag up to ``maxlags``: a series whose dependence sits at lag 2 or beyond
        can show no loss here while the HAC standard error can differ materially
        from the naive one. The HAC statistic is the inference; this count only
        restates the lag-1 autocorrelation as a number of observations ("104
        days of evidence, worth about 61 independent observations under
        AR(1)").

        Returns n unchanged when rho is non-positive: negative autocorrelation
        would formally imply MORE independent information than observations, and
        claiming a bonus from it would be the same overstatement this framework
        exists to prevent, in the opposite direction.
        """
        if not np.isfinite(self.lag1_autocorr) or self.n_obs <= 0:
            return float(self.n_obs)
        rho = self.lag1_autocorr
        if rho <= 0.0:
            return float(self.n_obs)
        # Guard the pole at rho -> 1, where the formula diverges.
        rho = min(rho, 0.99)
        return float(self.n_obs * (1.0 - rho) / (1.0 + rho))

    @property
    def information_loss(self) -> float:
        """Share of the nominal sample lost to lag-1 dependence, under the AR(1)
        approximation of ``effective_n``; zero does not rule out dependence at
        higher lags."""
        if self.n_obs <= 0:
            return float("nan")
        return 1.0 - self.effective_n / self.n_obs

    @property
    def se_inflation(self) -> float:
        """Ratio of the HAC standard error to the naive one.

        The two standard errors can differ; the ratio shows by how much. It is
        not attributed here to any particular sign or lag of dependence.
        """
        if not np.isfinite(self.naive_se) or self.naive_se <= 0:
            return float("nan")
        return float(self.hac_se / self.naive_se)

    def to_dict(self) -> dict:
        return {
            "n_obs": self.n_obs,
            "mean": self.mean,
            "naive_se": self.naive_se,
            "naive_tstat": self.naive_tstat,
            "hac_se": self.hac_se,
            "hac_tstat": self.hac_tstat,
            "hac_pvalue": self.hac_pvalue,
            "se_inflation": self.se_inflation,
            "maxlags": self.maxlags,
            "lag1_autocorr": self.lag1_autocorr,
            "effective_n": self.effective_n,
            "information_loss": self.information_loss,
            "notes": list(self.notes),
        }


def auto_maxlags(n: int) -> int:
    """``floor(4 * (n/100)^(2/9))``, floored at 1."""
    if n <= 1:
        return 1
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _lag1_autocorr(x: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    a, b = x[:-1], x[1:]
    if a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def newey_west_tstat(
    series: pd.Series | np.ndarray,
    maxlags: int | None = None,
) -> SignificanceResult:
    """HAC-corrected test that the mean of ``series`` differs from zero.

    Implemented as an OLS regression of the series on a constant with
    ``cov_type='HAC'``. That formulation is used rather than a hand-rolled
    Newey-West sum because the small-sample and bandwidth conventions in
    ``statsmodels`` are documented and standard -- reproducibility by a reviewer
    matters more here than avoiding a dependency.
    """
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    n = x.size
    notes: list[str] = []

    if n < MIN_OBS_FOR_TEST:
        return SignificanceResult(
            n_obs=n,
            mean=float(x.mean()) if n else float("nan"),
            naive_se=float("nan"),
            naive_tstat=float("nan"),
            hac_se=float("nan"),
            hac_tstat=float("nan"),
            hac_pvalue=float("nan"),
            maxlags=0,
            lag1_autocorr=float("nan"),
            notes=[f"only {n} observations; need >= {MIN_OBS_FOR_TEST} for inference"],
        )

    lags = auto_maxlags(n) if maxlags is None else int(maxlags)
    if lags >= n:
        lags = max(1, n - 2)
        notes.append(f"maxlags reduced to {lags} (must be < number of observations)")

    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    naive_se = sd / np.sqrt(n) if sd > 0 else float("nan")
    naive_t = mean / naive_se if naive_se and np.isfinite(naive_se) else float("nan")

    try:
        import statsmodels.api as sm

        model = sm.OLS(x, np.ones((n, 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
        )
        hac_se = float(model.bse[0])
        hac_t = float(model.tvalues[0])
        hac_p = float(model.pvalues[0])
    except Exception as exc:  # keep the audit runnable if statsmodels is absent
        notes.append(f"HAC estimation unavailable ({type(exc).__name__}); naive SE only")
        hac_se = hac_t = hac_p = float("nan")

    ac1 = _lag1_autocorr(x)
    if np.isfinite(ac1) and ac1 > 0.2:
        n_eff = n * (1.0 - min(ac1, 0.99)) / (1.0 + min(ac1, 0.99))
        notes.append(
            f"lag-1 autocorrelation {ac1:.2f}: under an AR(1) approximation, {n} "
            f"observations carry about {n_eff:.0f} independent observations' "
            "worth of information. This is a lag-1 heuristic, not a statement "
            "about the naive or HAC standard error; dependence beyond lag 1 is "
            "not in this count"
        )

    return SignificanceResult(
        n_obs=n,
        mean=mean,
        naive_se=naive_se,
        naive_tstat=naive_t,
        hac_se=hac_se,
        hac_tstat=hac_t,
        hac_pvalue=hac_p,
        maxlags=lags,
        lag1_autocorr=ac1,
        notes=notes,
    )
