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
from .panel import LABEL as LABEL_COL
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


def generate_drifting_panel(
    spec: SyntheticSpec | None = None,
    *,
    drift: float = 1.5,
    noise_sd: float = 0.5,
    seed: int = 1,
) -> Panel:
    """Panel whose feature-to-label relationship changes over time.

    Two competing features, with the true weight shifting from the first to the
    second across the sample:

        label = (1 - drift*t) * f_a + (drift*t) * f_b + noise

    where ``t`` runs from 0 to 1. ``drift=0`` gives a stationary relationship.

    This is the setting where random splitting leaks. With a stationary
    relationship and a low-capacity model there is nothing for random assignment
    to exploit -- training on a random subset and training on the past yield the
    same coefficients. Once the relationship drifts, a random split hands the
    model training rows drawn from the *test period's* regime, so it fits a
    relationship that a forecaster standing at the fold boundary could not have
    known. That is the mechanism the protocol audit exists to quantify, and the
    stationary case is the control that shows the audit does not flag splitting
    per se.

    Features are carried on the panel as ``f_a`` and ``f_b`` so the audit can
    refit under each protocol. ``prediction`` is filled with the label mean,
    since this panel is an input to refitting rather than a finished result.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(seed)

    base, _ = generate_panel(spec, skill=0.3, level_leak=1.0)
    d = base.data.sort_values([ENTITY_COL, DATE_COL], kind="mergesort").copy()

    dates = pd.DatetimeIndex(sorted(d[DATE_COL].unique()))
    position = d[DATE_COL].map({dt: i / max(1, len(dates) - 1) for i, dt in enumerate(dates)})

    d["f_a"] = rng.normal(size=len(d))
    d["f_b"] = rng.normal(size=len(d))
    weight_a = 1.0 - drift * position
    weight_b = drift * position
    d[LABEL_COL] = (
        weight_a * d["f_a"] + weight_b * d["f_b"] + rng.normal(0.0, noise_sd, len(d))
    )
    # A real prediction, fitted on the training period only and applied
    # throughout. The panel needs one: the protocol audit refits from the
    # features, but every other module scores a finished prediction, and a
    # constant placeholder would make their correlations undefined and leave the
    # report mostly empty.
    #
    # Fitting on training data alone is what makes this an honest baseline for
    # the drift story: the model learns the early regime and is then applied to a
    # later one, which is exactly the situation walk-forward validation is
    # designed to reveal and random splitting is designed (accidentally) to hide.
    train_mask = d[DATE_COL] <= base.train_end
    x_train = np.column_stack(
        [np.ones(int(train_mask.sum())), d.loc[train_mask, ["f_a", "f_b"]].to_numpy(float)]
    )
    y_train = d.loc[train_mask, LABEL_COL].to_numpy(float)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    x_all = np.column_stack([np.ones(len(d)), d[["f_a", "f_b"]].to_numpy(float)])
    d[PRED_COL] = x_all @ coef

    return Panel(
        data=d.reset_index(drop=True),
        train_end=base.train_end,
        label_name="synthetic_drifting_relationship",
    )


def generate_return_panel(
    spec: SyntheticSpec | None = None,
    *,
    lookahead: float = 0.0,
    skill: float = 0.15,
    seed: int = 5,
) -> Panel:
    """Panel with a return series, for the execution-timing audit.

    ``lookahead`` controls the defect. At 0 the prediction forecasts the NEXT
    period's return honestly. At 1 it is built from the CURRENT period's return
    -- the signature of a signal computed from a close and assumed to trade at
    that same close.

    The two produce very different decay profiles, which is the whole point:

    * honest -- scores modestly at every lag, decaying gently
    * look-ahead -- scores spectacularly at lag 0 and collapses at lag 1

    Returns are the label here, unlike the volatility-style target used
    elsewhere. Execution timing only has meaning against something tradeable, and
    a sign-constant target would make the decay profile uninterpretable.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(seed)

    entities = [f"R{i:04d}" for i in range(spec.n_entities)]
    dates = pd.bdate_range("2021-01-04", periods=spec.n_dates)

    # Returns are near-independent across time, as returns are, which is what
    # makes the lag test sharp: there is no persistence to blur the profile.
    returns = rng.normal(0.0, 0.02, size=(spec.n_dates, spec.n_entities))

    prediction = np.empty_like(returns)
    for t in range(spec.n_dates):
        forward = returns[t + 1] if t + 1 < spec.n_dates else np.zeros(spec.n_entities)
        honest = skill * forward
        peek = returns[t]  # the return the signal should not yet know
        prediction[t] = (
            (1.0 - lookahead) * honest
            + lookahead * peek
            + rng.normal(0.0, 0.02 * (1.0 - 0.5 * lookahead), size=spec.n_entities)
        )

    n = spec.n_dates * spec.n_entities
    frame = pd.DataFrame(
        {
            "entity_id": np.tile(entities, spec.n_dates),
            "event_date": np.repeat(dates.values, spec.n_entities),
            "prediction": prediction.reshape(n),
            "label": returns.reshape(n),
            "forward_return": returns.reshape(n),
        }
    )
    train_end = dates[int(spec.n_dates * spec.train_fraction) - 1]
    return Panel.from_frame(frame, train_end=train_end, label_name="synthetic_return")


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
