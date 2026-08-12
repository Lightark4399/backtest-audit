"""Incremental predictive power over a baseline, via partial correlation.

The question: given that a naive baseline ``b`` already correlates with the label
``y``, how much does the model prediction ``x`` add?

Two tempting answers, both wrong
--------------------------------
1. **Difference of correlations**, ``corr(x, y) - corr(b, y)``. Not a
   correlation, unbounded in interpretation, and not comparable across dates or
   datasets.

2. **Correlation of differences**, ``corr(x - b, y - b)``. This looks like
   "removing the baseline from both sides", but the shared ``-b`` term is common
   to both arguments and induces correlation of its own. When ``Var[b]`` is large
   relative to the residuals the statistic is dominated by ``b`` and can exceed
   ``corr(x, y)`` -- incoherent for something described as an increment.

The right statistic is the **partial correlation** of ``x`` and ``y``
controlling for ``b``: the correlation between the parts of each that are
linearly unexplained by ``b``.

    r_xy·b = (r_xy - r_xb · r_yb) / sqrt((1 - r_xb²)(1 - r_yb²))

It is a genuine correlation in [-1, 1], reduces to ``r_xy`` when ``b`` is
uninformative, and goes to 0 when the model adds nothing beyond ``b``.

The subtlety that makes this module non-trivial: errors-in-variables
--------------------------------------------------------------------
Partial correlation removes only what the control *actually measures*. Every
baseline in this framework is a **noisy proxy** for the stable entity level:

* ``persistence`` is a single observation, so it carries a full period of
  transient deviation and noise alongside the level.
* ``ewma`` averages a handful of observations: less noise, still some.
* ``per_entity_train_mean`` averages the whole training period, so the least --
  but not zero, since the deviation process is autocorrelated and its sample
  mean converges slowly.

Controlling for a noisy proxy leaves residual level information in both
residuals, and the partial correlation then credits the model for level knowledge
it obtained for free. This is not a coding error; it is the classic
attenuation-from-measurement-error problem, and it is large enough to matter. On a
synthetic panel built with *exactly zero* genuine skill and perfect level
knowledge, the naive partial correlation reads about +0.22 controlling for
``persistence`` and +0.10 controlling for ``per_entity_train_mean`` -- entirely
spurious increment.

The fix, and why it works
-------------------------
Remove the entity level from all three series **by demeaning each with its own
training-period mean** before computing the partial correlation. Demeaning does
not suffer the proxy problem: each series' level is estimated from that series
itself, so the estimate is unbiased for the quantity being removed rather than a
mismeasured stand-in imported from elsewhere.

Composed this way the statistic behaves correctly on the same synthetic panels:
approximately 0.00 at zero skill, rising monotonically with true skill. Hence
``demean=True`` is the default.

One consequence worth stating: when the control *is* the level estimate
(``per_entity_train_mean``), demeaning leaves it identically zero, its variance
vanishes, and the partial correlation is correctly reported as **undefined** --
one cannot control for the level twice. The framework says so rather than
fabricating a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..panel import DATE, ENTITY, LABEL, PRED, Panel
from .baselines import Baseline
from .ic import MIN_CROSS_SECTION, ICSeries, demean_by_train_mean

# Below this the control explains essentially all variance and no residual
# variation remains to correlate.
MIN_RESIDUAL_VARIANCE = 1e-10


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < MIN_CROSS_SECTION:
        return None
    sa, sb = a.std(), b.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa <= 0 or sb <= 0:
        return None
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else None


def partial_correlation(x: np.ndarray, y: np.ndarray, b: np.ndarray) -> float | None:
    """Partial correlation of ``x`` and ``y`` controlling for ``b``.

    Returns None when undefined: too few points, a degenerate input, or a control
    that explains virtually all variance in either series.
    """
    r_xy = _pearson(x, y)
    r_xb = _pearson(x, b)
    r_yb = _pearson(y, b)
    if r_xy is None or r_xb is None or r_yb is None:
        return None

    denom_sq = (1.0 - r_xb**2) * (1.0 - r_yb**2)
    if denom_sq <= MIN_RESIDUAL_VARIANCE:
        return None

    val = (r_xy - r_xb * r_yb) / np.sqrt(denom_sq)
    if not np.isfinite(val):
        return None

    # Floating point can push a boundary case a hair outside [-1, 1]; clamp
    # within tolerance, otherwise treat as undefined rather than emit an
    # impossible correlation.
    if val > 1.0:
        return 1.0 if val < 1.0 + 1e-8 else None
    if val < -1.0:
        return -1.0 if val > -1.0 - 1e-8 else None
    return float(val)


def incremental_ic(
    panel: Panel,
    baseline: Baseline,
    scope: str = "test",
    rank: bool = True,
    demean: bool = True,
) -> ICSeries:
    """Per-date partial correlation of prediction and label, controlling for a baseline.

    Parameters
    ----------
    rank:
        Convert all three series to within-date average ranks first, giving a
        rank-based partial correlation. Default True: once the level is removed
        the remaining deviations are heavy-tailed enough that a single extreme
        entity can otherwise dominate a day's estimate.
    demean:
        Remove each series' training-period per-entity mean before computing the
        partial correlation. Default True -- see the module docstring: without it
        the statistic is biased upward by measurement error in the control and
        reports substantial increment even for a model with no skill at all.
        ``demean=False`` is retained only so that bias can be demonstrated (the
        demo does exactly that) and is flagged in the result metadata.
    """
    work = panel.data.copy()
    work["_baseline"] = baseline.build(panel).reindex(work.index)

    if demean:
        te = panel.require_train_end("incremental_ic(demean=True)")

        # Demean prediction and label through the shared helper, so this module
        # and demeaned_ic can never disagree about what "demeaned" means.
        tmp = Panel(data=work, train_end=te, label_name=panel.label_name)
        view, diag = demean_by_train_mean(tmp)

        # The control gets the same treatment using its own training mean.
        base_train_mean = work.loc[work[DATE] <= te].groupby(ENTITY)["_baseline"].mean()
        view = view.copy()
        view["_baseline_dm"] = view["_baseline"] - view[ENTITY].map(base_train_mean)

        pred_col, label_col, base_col = (
            "prediction_demeaned",
            "label_demeaned",
            "_baseline_dm",
        )
        diag_meta = dict(diag)
    else:
        eval_dates = panel.evaluation_view(scope)[DATE].unique()
        view = work.loc[work[DATE].isin(eval_dates)].copy()
        pred_col, label_col, base_col = PRED, LABEL, "_baseline"
        diag_meta = {}

    n_dates_total = int(view[DATE].nunique())

    dates, vals, sizes = [], [], []
    n_undef = n_small = 0

    for date, g in view.groupby(DATE, sort=True):
        # All three correlations must use the SAME rows. A baseline is undefined
        # for an entity's first observation, so this subset is smaller than the
        # full cross-section; mixing populations would produce a partial
        # correlation corresponding to no actual sample.
        g = g.loc[g[base_col].notna() & g[pred_col].notna() & g[label_col].notna()]
        if len(g) < MIN_CROSS_SECTION:
            n_small += 1
            continue

        x = g[pred_col].to_numpy(dtype=float)
        y = g[label_col].to_numpy(dtype=float)
        b = g[base_col].to_numpy(dtype=float)

        if rank:
            x = pd.Series(x).rank(method="average").to_numpy()
            y = pd.Series(y).rank(method="average").to_numpy()
            b = pd.Series(b).rank(method="average").to_numpy()

        val = partial_correlation(x, y, b)
        if val is None:
            n_undef += 1
            continue
        dates.append(date)
        vals.append(val)
        sizes.append(len(g))

    meta = {
        "control": baseline.name,
        "rank_based": rank,
        "demeaned": demean,
        "scope": scope,
        "statistic": "partial_correlation",
        **diag_meta,
    }
    if not demean:
        meta["warning"] = (
            "demean=False: biased upward by measurement error in the control; "
            "not a valid increment estimate"
        )
    if n_dates_total and n_undef == n_dates_total:
        meta["note"] = (
            "undefined on every date -- the control has no residual variation "
            "after demeaning (cannot control for the entity level twice)"
        )

    return ICSeries(
        name=f"incremental_ic_vs_{baseline.name}",
        values=pd.Series(vals, index=pd.DatetimeIndex(dates), dtype=float),
        n_obs=pd.Series(sizes, index=pd.DatetimeIndex(dates), dtype=float),
        n_dates_total=n_dates_total,
        n_dates_undefined=n_undef,
        n_dates_too_small=n_small,
        meta=meta,
    )
