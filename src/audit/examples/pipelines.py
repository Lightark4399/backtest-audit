"""A pipeline with deliberate defects, each switchable.

Every defect below is one people actually ship. They are implemented in the way
they are usually written -- which is to say, in the way that looks reasonable in
review -- rather than as caricatures, because a demonstration that only catches
obviously-wrong code proves nothing about catching real mistakes.

The four defects:

``full_sample_scaling``
    Standardise features using statistics computed over the whole sample,
    including the evaluation period. Nearly universal, because the natural place
    to put a scaler is next to the data loading, before anyone thinks about
    splits.

``contemporaneous_feature``
    Include a feature computed from the same day as the label. This is the pure
    look-ahead: the feature is not available when the forecast is claimed to
    have been made.

``full_sample_tuning``
    Choose the hyper-parameter by scoring on the evaluation period. The model
    class is honest; the *selection* is what leaks.

``survivorship``
    Restrict the universe to entities that survive to the end of the sample.
    Distinct from the others in that it does not leak a value into any single
    row -- it biases which rows exist at all.

``misaligned_labels``
    Shift each entity's predictions one date out of place -- a plain off-by-one,
    the kind produced by a lag applied in the wrong direction. Its effect here is
    instructive and is discussed below.

Which defects which module catches
----------------------------------
Worth stating plainly, because a demonstration that implied the framework caught
everything would be dishonest:

* ``misaligned_labels`` is NOT caught by the alignment audit in this pipeline,
  and the reason is worth understanding. The features here are lagged labels, so
  after the shift the prediction for date t is a function of the label at t --
  the off-by-one has turned into contemporaneous leakage. The pairing the
  alignment audit examines is then genuinely "correct": the prediction really
  does match the label it is scored against. What is wrong is that it could not
  have been computed in time, which is a question about vintage, not pairing.
  The alignment audit detects misalignment when the prediction is NOT itself
  built from lagged labels (``tests/test_alignment.py`` covers that case with a
  panel constructed for it); for autoregressive pipelines an off-by-one presents
  as leakage and needs the point-in-time module instead.
* ``contemporaneous_feature`` is NOT caught by the alignment audit, and this is
  correct behaviour rather than a gap in it: the prediction really does match
  the label it is scored against, so the pairing is sound. What is wrong is that
  the prediction could not have been made when it claims to have been -- a
  question about data vintage, which the point-in-time module addresses.
* ``survivorship`` biases which rows exist, so no row-level check can see it; it
  needs a universe reconstruction from listing and delisting dates.
* ``full_sample_scaling`` and ``full_sample_tuning`` inflate the score without
  breaking any single row's validity. They show up as a gap between this
  pipeline and the clean one, which is why the two are run side by side.

The pipeline is a per-entity ridge regression on lagged features. The modelling
is intentionally plain: the point of the example is the surrounding protocol,
and a sophisticated model would only distract from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..panel import DATE, ENTITY, LABEL, PRED, Panel
from ..synthetic import SyntheticSpec, generate_panel


@dataclass(frozen=True)
class LeakConfig:
    """Which defects are active. All False reproduces the clean pipeline."""

    full_sample_scaling: bool = True
    contemporaneous_feature: bool = True
    full_sample_tuning: bool = True
    survivorship: bool = True
    misaligned_labels: bool = False

    @classmethod
    def clean(cls) -> LeakConfig:
        return cls(False, False, False, False, False)

    @classmethod
    def misaligned(cls) -> LeakConfig:
        """Only the off-by-one, so the alignment audit can be seen working alone."""
        return cls(False, False, False, False, True)

    def active(self) -> list[str]:
        return [k for k, v in self.__dict__.items() if v]


def _build_features(panel_data: pd.DataFrame, contemporaneous: bool) -> pd.DataFrame:
    """Lagged features from the label history.

    With ``contemporaneous=False`` every feature ends at the previous
    observation: ``shift(1)`` is applied BEFORE the rolling window, so the
    current label cannot enter its own predictor. This mirrors the SQL frame
    ending at ``1 PRECEDING``.

    With ``contemporaneous=True`` one feature uses the current row. That single
    change is enough to make the result look excellent while being unusable.
    """
    d = panel_data.sort_values([ENTITY, DATE], kind="mergesort").copy()
    g = d.groupby(ENTITY, sort=False)[LABEL]

    lagged = g.shift(1)
    d["f_lag1"] = lagged
    d["f_mean5"] = lagged.groupby(d[ENTITY], sort=False).transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    d["f_mean20"] = lagged.groupby(d[ENTITY], sort=False).transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    d["f_sd20"] = lagged.groupby(d[ENTITY], sort=False).transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )

    if contemporaneous:
        # THE LEAK: a quantity measured on the SAME day as the label. In real
        # pipelines this is rarely the label itself -- it is a correlated
        # same-day measurement whose timestamp was assumed to be available
        # earlier than it is (an end-of-day figure treated as known at the open,
        # a vendor field stamped with the trade date but published overnight).
        # Modelled here as a noisy same-day proxy rather than a copy of the
        # label, so the resulting inflation is large but not absurd -- a
        # pipeline reporting a perfect score would be caught by inspection, and
        # would prove nothing about catching realistic mistakes.
        rng = np.random.default_rng(11)
        d["f_today"] = d[LABEL] + rng.normal(0.0, d[LABEL].std() * 0.8, size=len(d))

    return d


def _feature_columns(d: pd.DataFrame) -> list[str]:
    return [c for c in d.columns if c.startswith("f_")]


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge with an unpenalised intercept, solved in closed form."""
    n = x.shape[1]
    xd = np.column_stack([np.ones(len(x)), x])
    reg = np.eye(n + 1) * alpha
    reg[0, 0] = 0.0  # never penalise the intercept
    try:
        return np.linalg.solve(xd.T @ xd + reg, xd.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(xd.T @ xd + reg, xd.T @ y, rcond=None)[0]


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ w


def run_pipeline(
    config: LeakConfig,
    spec: SyntheticSpec | None = None,
    skill: float = 0.35,
    alpha_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
) -> Panel:
    """Fit per-entity ridge models and return a Panel of out-of-sample predictions.

    The generated data has genuine but modest skill available (``skill=0.35``),
    so a correct pipeline finds something real and small. That is the honest
    baseline against which the leaks' inflation is measured -- if the clean
    version found nothing, the comparison would only show noise.
    """
    spec = spec or SyntheticSpec()
    source, _ = generate_panel(spec, skill=skill, level_leak=1.0)
    data = source.data.copy()
    train_end = source.train_end

    if config.survivorship:
        # THE LEAK: keep only entities present on the final date. In real data
        # this is what happens when the universe is pulled from a current
        # constituent list and history is backfilled -- the failures are simply
        # absent. Here the synthetic panel is balanced, so we simulate it by
        # dropping the entities whose evaluation-period labels are worst, which
        # is the effect survivorship has: the survivors are the ones that did
        # well.
        test = data.loc[data[DATE] > train_end]
        rank = test.groupby(ENTITY)[LABEL].mean().rank(pct=True)
        survivors = set(rank[rank > 0.2].index)
        data = data.loc[data[ENTITY].isin(survivors)].copy()

    feats = _build_features(data, contemporaneous=config.contemporaneous_feature)
    cols = _feature_columns(feats)
    feats = feats.dropna(subset=cols + [LABEL]).copy()

    if config.full_sample_scaling:
        # THE LEAK: mean and sd computed over the whole sample, evaluation period
        # included. Written the way it usually is -- once, near the top, before
        # any split exists to remind you not to.
        mu, sd = feats[cols].mean(), feats[cols].std().replace(0.0, 1.0)
        feats[cols] = (feats[cols] - mu) / sd
    else:
        tr = feats.loc[feats[DATE] <= train_end]
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0.0, 1.0)
        feats[cols] = (feats[cols] - mu) / sd

    preds: list[pd.DataFrame] = []
    for _entity, g in feats.groupby(ENTITY, sort=False):
        g = g.sort_values(DATE, kind="mergesort")
        tr = g.loc[g[DATE] <= train_end]
        te = g.loc[g[DATE] > train_end]
        if len(tr) < 30 or te.empty:
            continue

        xtr, ytr = tr[cols].to_numpy(float), tr[LABEL].to_numpy(float)
        xte = te[cols].to_numpy(float)

        if config.full_sample_tuning:
            # THE LEAK: alpha selected by scoring on the evaluation period. The
            # model is only ever FIT on training data, which is what makes this
            # easy to defend in review -- but the choice of alpha has seen the
            # test labels, so the reported score is optimistic.
            yte = te[LABEL].to_numpy(float)
            best, best_err = alpha_grid[0], np.inf
            for a in alpha_grid:
                err = np.mean(np.abs(_predict(xte, _fit_ridge(xtr, ytr, a)) - yte))
                if err < best_err:
                    best, best_err = a, err
            alpha = best
        else:
            # Honest alternative: a time-ordered hold-out carved from the tail of
            # the training period. Never touches evaluation labels.
            cut = int(len(tr) * 0.8)
            if cut < 20 or len(tr) - cut < 5:
                alpha = 1e-2
            else:
                xa, ya = xtr[:cut], ytr[:cut]
                xv, yv = xtr[cut:], ytr[cut:]
                best, best_err = alpha_grid[0], np.inf
                for a in alpha_grid:
                    err = np.mean(np.abs(_predict(xv, _fit_ridge(xa, ya, a)) - yv))
                    if err < best_err:
                        best, best_err = a, err
                alpha = best

        w = _fit_ridge(xtr, ytr, alpha)
        out = te[[ENTITY, DATE, LABEL]].copy()
        out[PRED] = _predict(xte, w)
        preds.append(out)

    if not preds:
        raise RuntimeError("no entity produced predictions; check the spec")

    result = pd.concat(preds, ignore_index=True)

    if config.misaligned_labels:
        # THE DEFECT: each prediction is paired with the NEXT date's label. This
        # is what a lag applied in the wrong direction produces, and unlike the
        # other defects it leaves a signature the alignment audit can detect.
        result = result.sort_values([ENTITY, DATE], kind="mergesort")
        result[PRED] = result.groupby(ENTITY, sort=False)[PRED].shift(-1)
        result = result.loc[result[PRED].notna()].reset_index(drop=True)

    # Training rows are re-attached so the audit can compute per-entity training
    # means. They carry no predictions, so they are given the entity's training
    # mean label as a placeholder prediction: this affects only the training-mean
    # of the prediction series, never an evaluation-period metric.
    tr_rows = feats.loc[feats[DATE] <= train_end, [ENTITY, DATE, LABEL]].copy()
    tr_mean = tr_rows.groupby(ENTITY)[LABEL].transform("mean")
    tr_rows[PRED] = tr_mean
    combined = pd.concat([tr_rows, result], ignore_index=True)

    return Panel.from_frame(
        combined[[ENTITY, DATE, PRED, LABEL]],
        train_end=train_end,
        label_name="synthetic_persistent_target",
    )


def run_leaky(**kwargs) -> Panel:
    """All four defects active."""
    return run_pipeline(LeakConfig(), **kwargs)


def run_clean(**kwargs) -> Panel:
    """Same modelling, no defects."""
    return run_pipeline(LeakConfig.clean(), **kwargs)


def run_misaligned(**kwargs) -> Panel:
    """Clean modelling with a single off-by-one.

    Included to show what an off-by-one looks like in an autoregressive pipeline:
    the score rises sharply, yet the alignment audit passes, because the defect
    has become a vintage problem rather than a pairing problem. See the module
    docstring.
    """
    return run_pipeline(LeakConfig.misaligned(), **kwargs)
