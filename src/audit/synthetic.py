"""Synthetic panel generator with known ground truth.

Two uses, both essential:

1. **Testing.** Assertions about a metric need cases whose correct answer is
   known analytically. Real market data cannot serve: nobody knows the true
   incremental predictive power in a real panel, so a test against it could only
   assert that today's output matches yesterday's, which locks in bugs rather
   than catching them.

2. **Offline demo.** The demo must run in CI and on any machine without network
   access or API credentials. A generator makes the demo hermetic and its output
   deterministic given a seed.

The generative model is built to reproduce the specific structure this framework
is about: a persistent target whose cross-sectional variance is dominated by
stable entity levels.

    log_target(i, t) = level(i) + persistent(i, t) + noise(i, t)

* ``level(i)`` is drawn once per entity and never changes. It is the free
  component: any predictor that recovers it scores well on raw IC while knowing
  nothing about dynamics.
* ``persistent(i, t)`` is an AR(1) process. It is the *learnable* component --
  the part a real model could forecast from features.
* ``noise(i, t)`` is unforecastable by construction and caps the attainable
  skill.

Because the decomposition is explicit, a caller can construct a prediction with
a chosen amount of genuine skill (``skill``) and a chosen amount of free level
information (``level_leak``) and then check that the framework attributes them
correctly. That is the property the test suite exercises: high raw IC with
``skill=0`` must yield an incremental IC near zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .panel import DATE as DATE_COL
from .panel import ENTITY as ENTITY_COL
from .panel import PRED as PRED_COL
from .panel import Panel


@dataclass(frozen=True)
class SyntheticSpec:
    """Parameters of the generated panel.

    Defaults are chosen so the generated data exhibits the phenomenon of
    interest: ``level_sd`` (0.60) is large relative to ``persistent_sd`` (0.25),
    so entity levels dominate the cross-section, as they do for realized
    volatility.
    """

    n_entities: int = 120
    n_dates: int = 260
    level_sd: float = 0.60
    persistent_sd: float = 0.25
    ar1: float = 0.85
    noise_sd: float = 0.20
    n_groups: int = 4  # sector-like grouping, used by group decomposition
    seed: int = 7

    @property
    def train_fraction(self) -> float:
        return 0.6


def generate_panel(
    spec: SyntheticSpec | None = None,
    *,
    skill: float = 0.5,
    level_leak: float = 1.0,
    pred_noise_sd: float = 1.0,
) -> tuple[Panel, pd.DataFrame]:
    """Generate a panel plus the hidden components behind it.

    Parameters
    ----------
    skill:
        Weight on the true persistent deviation in the prediction. ``0`` means
        the prediction contains no information about dynamics; ``1`` means it
        sees the deviation exactly (before ``pred_noise_sd`` is added).
    level_leak:
        Weight on the entity level in the prediction. ``1`` means the prediction
        knows each entity's level perfectly -- which is realistic, since a
        per-entity intercept learns it from training data, and it is what makes
        raw IC high for free.
    pred_noise_sd:
        Scale of the noise added to the prediction, expressed as a multiple of
        ``persistent_sd``. Larger values degrade skill without changing the level
        component.

    Returns
    -------
    (panel, truth)
        ``truth`` carries the latent ``level``, ``persistent`` and ``noise``
        columns so tests can verify attribution against the actual generative
        decomposition.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)

    entities = [f"E{i:04d}" for i in range(spec.n_entities)]
    dates = pd.bdate_range("2021-01-04", periods=spec.n_dates)

    level = rng.normal(0.0, spec.level_sd, size=spec.n_entities)
    group = rng.integers(0, spec.n_groups, size=spec.n_entities)

    # AR(1) deviations, initialised at the stationary distribution so the first
    # dates are not systematically closer to zero than later ones (which would
    # make early cross-sections behave differently from late ones).
    stationary_sd = spec.persistent_sd / np.sqrt(max(1e-12, 1.0 - spec.ar1**2))
    dev = np.empty((spec.n_dates, spec.n_entities))
    dev[0] = rng.normal(0.0, stationary_sd, size=spec.n_entities)
    for t in range(1, spec.n_dates):
        dev[t] = spec.ar1 * dev[t - 1] + rng.normal(
            0.0, spec.persistent_sd, size=spec.n_entities
        )

    noise = rng.normal(0.0, spec.noise_sd, size=(spec.n_dates, spec.n_entities))
    label = level[None, :] + dev + noise

    pred_noise = rng.normal(
        0.0, pred_noise_sd * spec.persistent_sd, size=(spec.n_dates, spec.n_entities)
    )
    prediction = level_leak * level[None, :] + skill * dev + pred_noise

    n_rows = spec.n_dates * spec.n_entities
    frame = pd.DataFrame(
        {
            "entity_id": np.tile(entities, spec.n_dates),
            "event_date": np.repeat(dates.values, spec.n_entities),
            "prediction": prediction.reshape(n_rows),
            "label": label.reshape(n_rows),
            "group": np.tile(group, spec.n_dates),
        }
    )
    truth = pd.DataFrame(
        {
            "entity_id": np.tile(entities, spec.n_dates),
            "event_date": np.repeat(dates.values, spec.n_entities),
            "level": np.tile(level, spec.n_dates),
            "persistent": dev.reshape(n_rows),
            "noise": noise.reshape(n_rows),
        }
    )

    train_end = dates[int(spec.n_dates * spec.train_fraction) - 1]
    panel = Panel.from_frame(
        frame, train_end=train_end, label_name="synthetic_persistent_target"
    )
    return panel, truth


def generate_panel_with_delisting(
    spec: SyntheticSpec | None = None,
    *,
    skill: float = 0.4,
    delist_fraction: float = 0.25,
    delist_hardness: float = 0.6,
) -> tuple[Panel, pd.DataFrame]:
    """Panel where a subset of entities stops trading part-way through.

    ``delist_hardness`` controls the mechanism that makes survivorship a bias
    rather than merely attrition: entities selected for delisting have their
    predictable component scaled DOWN by this factor, so they are genuinely
    harder to forecast. Dropping them therefore inflates the score.

    That coupling is the whole point. Random attrition -- entities leaving for
    reasons unrelated to the target -- costs sample size but introduces no bias,
    and a survivorship audit run on randomly-attriting data should correctly
    report no gap. Making the mechanism explicit lets both cases be tested.

    Returns the panel plus a frame of listing/delisting dates suitable for
    loading into the ``entities`` table.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed + 991)

    panel, truth = generate_panel(spec, skill=skill, level_leak=1.0)
    data = panel.data.copy()
    entities = sorted(data[ENTITY_COL].unique())
    dates = pd.DatetimeIndex(sorted(data[DATE_COL].unique()))

    n_delist = int(len(entities) * delist_fraction)
    doomed = set(rng.choice(entities, size=n_delist, replace=False))

    # Delisting is spread across the EVALUATION period rather than the whole
    # history. An entity that leaves before evaluation begins contributes nothing
    # to either arm's score, so scattering delistings over the full panel would
    # dilute the very effect this generator exists to produce.
    train_idx = int(dates.searchsorted(panel.train_end))
    first, last = train_idx + 1, len(dates)
    delist_at = {e: dates[rng.integers(first, last)] for e in sorted(doomed)}

    # Doomed entities are harder to predict: their prediction is shrunk toward
    # the entity level, retaining the free component and losing the skilful one.
    level = data.groupby(ENTITY_COL)[PRED_COL].transform("mean")
    is_doomed = data[ENTITY_COL].isin(doomed)
    data.loc[is_doomed, PRED_COL] = (
        level[is_doomed]
        + (data.loc[is_doomed, PRED_COL] - level[is_doomed]) * delist_hardness
    )

    # Remove rows after each doomed entity's delisting date.
    keep = pd.Series(True, index=data.index)
    for e, d in delist_at.items():
        keep &= ~((data[ENTITY_COL] == e) & (data[DATE_COL] > d))
    data = data.loc[keep].reset_index(drop=True)

    meta = pd.DataFrame(
        {
            "entity_id": entities,
            "sector": [f"S{i % spec.n_groups}" for i in range(len(entities))],
            "listing_date": [dates[0]] * len(entities),
            "delisting_date": [delist_at.get(e) for e in entities],
        }
    )

    out = Panel(
        data=data, train_end=panel.train_end, label_name="synthetic_with_delisting"
    )
    return out, meta


def generate_perfect_panel(n_entities: int = 50, n_dates: int = 30) -> Panel:
    """Panel where prediction equals label exactly. Every IC must be 1.0."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    entities = [f"P{i:03d}" for i in range(n_entities)]
    y = rng.normal(size=(n_dates, n_entities))
    n = n_dates * n_entities
    frame = pd.DataFrame(
        {
            "entity_id": np.tile(entities, n_dates),
            "event_date": np.repeat(dates.values, n_entities),
            "prediction": y.reshape(n),
            "label": y.reshape(n),
        }
    )
    return Panel.from_frame(frame, train_end=dates[int(n_dates * 0.5)])


def generate_random_panel(n_entities: int = 60, n_dates: int = 120, seed: int = 3) -> Panel:
    """Panel where prediction is independent of label. Every IC must be ~0."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    entities = [f"R{i:03d}" for i in range(n_entities)]
    n = n_dates * n_entities
    frame = pd.DataFrame(
        {
            "entity_id": np.tile(entities, n_dates),
            "event_date": np.repeat(dates.values, n_entities),
            "prediction": rng.normal(size=n),
            "label": rng.normal(size=n),
        }
    )
    return Panel.from_frame(frame, train_end=dates[int(n_dates * 0.5)])
