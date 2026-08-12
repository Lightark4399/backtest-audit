"""Naive baselines.

The purpose of a baseline here is adversarial: it is an attempt to reproduce the
model's reported score using no modelling at all. Whatever score a baseline
achieves is *free* -- available to anyone with the label history and no
predictive insight whatsoever -- and must be subtracted, conceptually, from any
claim made about the model.

Every baseline obeys the same information constraint as the model: the value
predicted for ``(entity, event_date)`` uses only labels dated strictly earlier
than ``event_date``. A baseline that peeked would understate the model's
increment, which is the opposite of the conservative error to make, so the
constraint is enforced by construction (via shifting) rather than by convention.

Four baselines, chosen to span the ways a persistent target leaks a free score:

``persistence``
    Last observed label for the entity. Captures short-horizon autocorrelation.

``per_entity_train_mean``
    The entity's average label over the training period. Captures the stable
    cross-entity level and nothing else -- it is constant in time per entity.
    This is the baseline a per-entity intercept reproduces for free, and is
    usually the most damaging comparison for a model reporting a high raw IC.

``ewma``
    Exponentially weighted mean of past labels. A stronger, more realistic
    persistence baseline; a model that cannot beat it has little practical value.

``cross_sectional_mean``
    That date's average label across entities. Constant within each
    cross-section by construction, hence its cross-sectional IC is *undefined*
    rather than zero -- it provides no ranking information at all. It is
    included as a control that verifies the metric layer reports undefined
    correlations honestly instead of silently coercing them to 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..panel import DATE, ENTITY, LABEL, Panel


@dataclass(frozen=True)
class Baseline:
    """A named naive predictor.

    ``build`` maps a Panel to a Series aligned on ``panel.data.index`` holding
    the baseline's prediction for every row (NaN where undefined, e.g. an
    entity's first observation has no prior label).
    """

    name: str
    description: str
    build: Callable[[Panel], pd.Series]
    is_constant_cross_section: bool = False


def _persistence(panel: Panel) -> pd.Series:
    """Previous observed label within each entity.

    ``groupby.shift(1)`` on a frame sorted by (entity, date) takes the entity's
    immediately preceding observation. Note this is the previous *observed*
    date, not the previous calendar date: if an entity is missing for a day, its
    last available label is used. That is the information a forecaster would
    actually have had, so it is the correct behaviour -- but it does mean the
    effective lag varies, which the report notes.
    """
    d = panel.data
    return d.groupby(ENTITY, sort=False)[LABEL].shift(1)


def _per_entity_train_mean(panel: Panel) -> pd.Series:
    """The entity's training-period mean label, broadcast to all its rows.

    Uses training data only -- see ``ic.demean_by_train_mean`` for why a
    full-sample mean would be leakage.
    """
    mu = panel.per_entity_train_mean(LABEL)
    return panel.data[ENTITY].map(mu)


def _make_ewma(halflife: float) -> Callable[[Panel], pd.Series]:
    def _ewma(panel: Panel) -> pd.Series:
        d = panel.data

        # shift(1) BEFORE the EWMA so the current label never enters its own
        # prediction. Order matters: applying ewm() first and shifting the
        # result would also work, but shifting the input makes the information
        # constraint syntactically obvious at the point of use.
        def _one(s: pd.Series) -> pd.Series:
            return s.shift(1).ewm(halflife=halflife, min_periods=1).mean()

        return d.groupby(ENTITY, sort=False)[LABEL].transform(_one)

    return _ewma


def _cross_sectional_mean(panel: Panel) -> pd.Series:
    """That date's mean label across entities, using only prior information.

    The same-day cross-sectional mean would embed the contemporaneous labels of
    all other entities -- information not available before the close. To keep
    the baseline honest it uses the *previous* date's cross-sectional mean.
    Either way the series is constant within a cross-section, so its IC is
    undefined; the shift matters only so that no baseline in this module can be
    accused of peeking.
    """
    d = panel.data
    daily = d.groupby(DATE)[LABEL].mean().sort_index().shift(1)
    return d[DATE].map(daily)


def default_baselines(ewma_halflife: float = 5.0) -> list[Baseline]:
    """The standard baseline set.

    ``ewma_halflife`` is in observations, not calendar days, and is configurable
    because the right persistence horizon is target-dependent -- a half-life
    tuned for daily realized volatility would be wrong for, say, monthly volume.
    """
    return [
        Baseline(
            name="persistence",
            description="last observed label for the entity",
            build=_persistence,
        ),
        Baseline(
            name="per_entity_train_mean",
            description="entity's mean label over the training period",
            build=_per_entity_train_mean,
        ),
        Baseline(
            name="ewma",
            description=f"EWMA of past labels (halflife={ewma_halflife:g} obs)",
            build=_make_ewma(ewma_halflife),
        ),
        Baseline(
            name="cross_sectional_mean",
            description="previous date's cross-sectional mean label",
            build=_cross_sectional_mean,
            is_constant_cross_section=True,
        ),
    ]


def build_baseline_panel(panel: Panel, baseline: Baseline) -> Panel:
    """Panel with the baseline substituted into the ``prediction`` column.

    Routing baselines through ``Panel.replace_prediction`` means they are scored
    by the exact same metric code as the model, so a discrepancy between model
    and baseline can never be an artefact of two different implementations.
    """
    values = baseline.build(panel)
    return panel.replace_prediction(values, label_name=panel.label_name)


def evaluate_baselines(
    panel: Panel,
    baselines: list[Baseline] | None = None,
    method: str = "spearman",
    scope: str = "test",
) -> pd.DataFrame:
    """Score every baseline on the same scope and metric as the model.

    Returns one row per baseline with its IC statistics. Baselines whose
    cross-section is constant will show ``n_dates_used == 0`` and an undefined
    mean; that is the correct, informative outcome and not a failure.
    """
    from .ic import cross_sectional_ic  # local import avoids a cycle

    baselines = baselines if baselines is not None else default_baselines()
    rows = []
    for b in baselines:
        try:
            bp = build_baseline_panel(panel, b)
            series = cross_sectional_ic(bp, method=method, scope=scope, name=b.name)
            rec = series.to_dict()
        except Exception as exc:  # a baseline that cannot be built is reported, not fatal
            rec = {
                "name": b.name,
                "mean": float("nan"),
                "std": float("nan"),
                "hit_rate": float("nan"),
                "n_dates_used": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rec["description"] = b.description
        rows.append(rec)

    return pd.DataFrame(rows).set_index("name")


def strongest_baseline(baseline_table: pd.DataFrame) -> str | None:
    """Name of the baseline with the highest mean IC, ignoring undefined ones.

    The *strongest* baseline is the right control for the partial-correlation
    increment: crediting a model for beating a weak baseline while a stronger
    free predictor exists would overstate its contribution.
    """
    valid = baseline_table["mean"].dropna()
    valid = valid[baseline_table.loc[valid.index, "n_dates_used"] > 0]
    if valid.empty:
        return None
    return str(valid.idxmax())
