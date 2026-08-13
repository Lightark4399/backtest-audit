"""Cross-sectional information coefficient (IC) metrics.

Three quantities are computed here, and the distinction between them is the
substance of this project:

``raw_ic``
    Per-date cross-sectional correlation between prediction and label, averaged
    over dates. This is what most backtests report.

``rank_ic``
    Same, using Spearman rank correlation. Robust to monotone transforms and to
    the heavy tails typical of financial targets.

``demeaned_ic``
    Per-date cross-sectional correlation after subtracting each entity's
    TRAINING-PERIOD mean from both series. This is the headline metric of the
    framework.

Why demeaning changes the question being asked
----------------------------------------------
Decompose a persistent target into a stable entity level plus a deviation::

    y(i, t) = m(i) + d(i, t)

For targets such as realized volatility or log volume, ``Var[m]`` across
entities is large relative to ``Var[d]``. Any prediction that merely recovers
``m(i)`` -- which a per-entity intercept does automatically -- will therefore
achieve a high cross-sectional correlation with ``y`` while carrying no
information about ``d`` at all. The raw IC is then close to a tautology: it
mostly certifies that entities which have historically been volatile are still
volatile.

Subtracting ``m(i)`` from both sides removes that free component and measures
correlation between predicted and realized *deviations*. It is the quantity a
user of the forecast actually needs, because the level is knowable without any
model.

The mean must come from the training period alone. Using a full-sample mean
would subtract a quantity computed partly from the evaluation labels, which
contaminates the metric with out-of-sample information -- the same class of
error this framework detects elsewhere. ``Panel.require_train_end`` enforces
this by raising rather than defaulting.

Degenerate cross-sections
-------------------------
A correlation is undefined when either series has zero variance within a date.
This is not an edge case to paper over: a prediction that is constant across
entities on a given day carries no cross-sectional ranking information, and the
honest report of its IC is "undefined", not "zero". Such dates are excluded from
the average and counted separately, so the report can state how often it
happened. (The cross-sectional-mean baseline in ``baselines.py`` is constant by
construction and exists precisely to exercise this path.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..panel import DATE, ENTITY, LABEL, PRED, Panel

MIN_CROSS_SECTION = 3  # below this, a correlation is noise, not a measurement


@dataclass
class ICSeries:
    """Per-date IC values plus bookkeeping about what could not be computed.

    Keeping the full daily series (rather than only its mean) is required for
    honest significance testing: the series is autocorrelated, so the standard
    error must be estimated from it directly (see ``significance.py``).
    """

    name: str
    values: pd.Series  # indexed by event_date, float
    n_obs: pd.Series  # cross-section size per date
    n_dates_total: int
    n_dates_undefined: int
    n_dates_too_small: int
    meta: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(self.values.mean()) if len(self.values) else float("nan")

    @property
    def std(self) -> float:
        # Sample std of the daily IC series; describes dispersion across days,
        # NOT the standard error of the mean (which needs an autocorrelation
        # correction -- see significance.newey_west_tstat).
        return float(self.values.std(ddof=1)) if len(self.values) > 1 else float("nan")

    @property
    def n_dates_used(self) -> int:
        return int(len(self.values))

    @property
    def hit_rate(self) -> float:
        """Fraction of dates with positive IC.

        Reported alongside the mean because a mean can be produced either by a
        small consistent edge or by a few large outlying days, and those two
        situations warrant different confidence.
        """
        if not len(self.values):
            return float("nan")
        return float((self.values > 0).mean())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "hit_rate": self.hit_rate,
            "n_dates_used": self.n_dates_used,
            "n_dates_total": self.n_dates_total,
            "n_dates_undefined": self.n_dates_undefined,
            "n_dates_too_small": self.n_dates_too_small,
            "mean_cross_section_size": (
                float(self.n_obs.mean()) if len(self.n_obs) else float("nan")
            ),
            **self.meta,
        }


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    """Correlation of two aligned arrays, or None if undefined.

    Returns None (rather than NaN) so callers must handle the undefined case
    explicitly instead of letting a NaN propagate into an average.
    """
    if x.size != y.size or x.size < MIN_CROSS_SECTION:
        return None
    if method == "spearman":
        # Average ranks for ties. Ranking first and then taking a Pearson
        # correlation is equivalent to Spearman and keeps a single code path.
        x = pd.Series(x).rank(method="average").to_numpy()
        y = pd.Series(y).rank(method="average").to_numpy()
    elif method != "pearson":
        raise ValueError(f"method must be 'pearson' or 'spearman'; got {method!r}")

    sx = x.std()
    sy = y.std()
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0.0 or sy <= 0.0:
        return None  # zero variance -> correlation undefined, not zero

    c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else None


def cross_sectional_ic(
    panel: Panel,
    method: str = "pearson",
    scope: str = "test",
    pred_col: str = PRED,
    label_col: str = LABEL,
    name: str | None = None,
) -> ICSeries:
    """Per-date cross-sectional IC between ``pred_col`` and ``label_col``."""
    view = panel.evaluation_view(scope)
    dates, ics, sizes = [], [], []
    n_undef = n_small = 0

    for date, g in view.groupby(DATE, sort=True):
        x = g[pred_col].to_numpy(dtype=float)
        y = g[label_col].to_numpy(dtype=float)
        if x.size < MIN_CROSS_SECTION:
            n_small += 1
            continue
        c = _corr(x, y, method)
        if c is None:
            n_undef += 1
            continue
        dates.append(date)
        ics.append(c)
        sizes.append(x.size)

    n_total = int(view[DATE].nunique())
    return ICSeries(
        name=name or f"{method}_ic",
        values=pd.Series(ics, index=pd.DatetimeIndex(dates), dtype=float),
        n_obs=pd.Series(sizes, index=pd.DatetimeIndex(dates), dtype=float),
        n_dates_total=n_total,
        n_dates_undefined=n_undef,
        n_dates_too_small=n_small,
        meta={"method": method, "scope": scope},
    )


def raw_ic(panel: Panel, scope: str = "test") -> ICSeries:
    """Pearson cross-sectional IC -- the conventionally reported number."""
    return cross_sectional_ic(panel, method="pearson", scope=scope, name="raw_ic")


def rank_ic(panel: Panel, scope: str = "test") -> ICSeries:
    """Spearman cross-sectional IC."""
    return cross_sectional_ic(panel, method="spearman", scope=scope, name="rank_ic")


def demean_by_train_mean(panel: Panel) -> tuple[pd.DataFrame, dict]:
    """Subtract each entity's training-period mean from prediction and label.

    Returns the demeaned frame (evaluation scope only) and diagnostics about
    entities that had to be dropped.

    Both series are demeaned by their OWN training mean, not by a shared one.
    The prediction's level may be biased relative to the label's -- a model can
    systematically over- or under-shoot -- and the question being asked is
    whether the two *deviations* co-move. Using the label's mean for both would
    fold the model's level bias into the deviation and answer a different
    question.
    """
    panel.require_train_end("demeaned_ic")

    pred_mean = panel.per_entity_train_mean(PRED)
    label_mean = panel.per_entity_train_mean(LABEL)

    view = panel.test_slice().copy()
    n_before = len(view)
    entities_before = view[ENTITY].nunique()

    view["_pred_mu"] = view[ENTITY].map(pred_mean)
    view["_label_mu"] = view[ENTITY].map(label_mean)

    # Entities absent from the training period have no mean to subtract. They
    # are dropped rather than demeaned with a cross-sectional stand-in: any
    # substitute would be a fabricated level, and silently mixing demeaned and
    # non-demeaned rows in one correlation is exactly the kind of quiet error
    # this tool exists to surface.
    usable = view["_pred_mu"].notna() & view["_label_mu"].notna()
    view = view.loc[usable].copy()

    view["prediction_demeaned"] = view[PRED] - view["_pred_mu"]
    view["label_demeaned"] = view[LABEL] - view["_label_mu"]

    diagnostics = {
        "rows_before_demean": n_before,
        "rows_after_demean": len(view),
        "entities_before": int(entities_before),
        "entities_dropped_no_train_history": int(entities_before - view[ENTITY].nunique()),
    }
    return view, diagnostics


def demeaned_ic(panel: Panel, method: str = "spearman") -> ICSeries:
    """Cross-sectional IC of training-mean-demeaned prediction vs label.

    Defaults to Spearman: after demeaning, the residual deviations are noisier
    and more heavy-tailed than the levels were, so a rank statistic is the more
    stable summary. Pearson is available via ``method``.
    """
    view, diag = demean_by_train_mean(panel)

    # Build a throwaway Panel over the demeaned columns so the identical
    # cross_sectional_ic code path is reused. train_end is carried through, and
    # scope='all' is correct here because `view` is already the test slice.
    tmp = Panel(data=view, train_end=panel.train_end, label_name=panel.label_name)
    series = cross_sectional_ic(
        tmp,
        method=method,
        scope="all",
        pred_col="prediction_demeaned",
        label_col="label_demeaned",
        name="demeaned_ic",
    )
    series.meta.update(diag)
    series.meta["demeaned"] = True
    return series
