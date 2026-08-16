"""Validation protocol audit: does the splitting scheme itself inflate the score?

Every other module examines the data or the prediction. This one examines the
*evaluation procedure*, and it is included because the most common way to inflate
a backtest is also the most innocuous-looking line of code in the pipeline:

    train_test_split(X, y, shuffle=True)

or its cross-validated cousin, ``KFold(shuffle=True)``.

Why random splitting leaks on a panel
-------------------------------------
Random assignment scatters observations across folds without regard to time. For
a persistent target, adjacent dates are near-duplicates: the label on Tuesday is
highly correlated with Monday's, and features built from lagged labels are
*literally computed from* overlapping data. When Monday lands in training and
Tuesday in test, the model has effectively seen the answer.

Nothing about this is visible in the data, the features, or the predictions. It
is a property of the procedure, so no row-level check can find it -- which is why
it needs a module of its own.

Two further leaks that purging and embargoing address
-----------------------------------------------------
Even an ordered split can leak at the boundary:

**Overlap.** A feature at date t may be built from a window ending at t, and a
label at date t-k may depend on data through t. Training rows whose information
window overlaps the test period must be dropped -- *purged* -- or the model has
been fit on data that describes the period it is being tested on.

**Serial dependence across the boundary.** Even without direct overlap, the last
training observation and the first test observation are adjacent in time and
therefore correlated through the persistence of the process. An *embargo* -- a
gap of a few periods between the end of training and the start of test -- removes
the most contaminated pairs.

What this module measures
-------------------------
The same model, the same data, scored under three protocols:

* ``random_kfold`` -- shuffled K-fold, ignoring time entirely
* ``walk_forward`` -- expanding window, train on the past, test on the future
* ``purged_walk_forward`` -- as above, with an embargo gap

The gap between the first and the last is the inflation attributable to the
protocol. On a persistent target it is usually large, and it is entirely
unearned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metrics.ic import MIN_CROSS_SECTION, _corr
from ..panel import DATE, ENTITY, LABEL, PRED, Panel

# Below this the protocols agree to within estimation noise.
MATERIAL_GAP = 0.02


@dataclass
class ProtocolResult:
    """Score achieved under one splitting protocol."""

    name: str
    description: str
    ic: float
    n_folds: int
    n_test_rows: int
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "ic": self.ic,
            "n_folds": self.n_folds,
            "n_test_rows": self.n_test_rows,
            **self.detail,
        }


@dataclass
class ProtocolComparison:
    """Comparison of random, ordered and purged protocols."""

    results: list[ProtocolResult]
    passed: bool | None
    verdict: str
    detail: dict = field(default_factory=dict)

    def by_name(self, name: str) -> ProtocolResult | None:
        return next((r for r in self.results if r.name == name), None)

    @property
    def inflation(self) -> float:
        """Random-split IC minus purged walk-forward IC."""
        a = self.by_name("random_kfold")
        b = self.by_name("purged_walk_forward")
        if a is None or b is None:
            return float("nan")
        return a.ic - b.ic

    @property
    def embargo_effect(self) -> float:
        """How much the embargo alone removes."""
        a = self.by_name("walk_forward")
        b = self.by_name("purged_walk_forward")
        if a is None or b is None:
            return float("nan")
        return a.ic - b.ic

    def to_dict(self) -> dict:
        return {
            "protocols": [r.to_dict() for r in self.results],
            "inflation": self.inflation,
            "embargo_effect": self.embargo_effect,
            "passed": self.passed,
            "verdict": self.verdict,
            **self.detail,
        }


def _fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], alpha: float
) -> np.ndarray | None:
    """Ridge with an unpenalised intercept. Deliberately the same for every protocol.

    Holding the estimator fixed is what makes the comparison about the protocol.
    Tuning inside each fold would introduce a second difference between arms and
    make the gap impossible to attribute.
    """
    if len(train) < len(feature_cols) + 2 or test.empty:
        return None
    x = np.column_stack([np.ones(len(train)), train[feature_cols].to_numpy(float)])
    y = train[LABEL].to_numpy(float)
    reg = np.eye(x.shape[1]) * alpha
    reg[0, 0] = 0.0
    try:
        w = np.linalg.solve(x.T @ x + reg, x.T @ y)
    except np.linalg.LinAlgError:
        return None
    xt = np.column_stack([np.ones(len(test)), test[feature_cols].to_numpy(float)])
    return xt @ w


def _score_predictions(frame: pd.DataFrame, method: str = "spearman") -> float:
    """Mean per-date cross-sectional IC over out-of-fold predictions.

    Scored per date rather than pooled for the same reason the rest of the
    framework does: a pooled correlation over stacked dates would be dominated by
    variation in the level between dates rather than by ranking within them.
    """
    ics = []
    for _, g in frame.groupby(DATE, sort=True):
        if len(g) < MIN_CROSS_SECTION:
            continue
        c = _corr(g[PRED].to_numpy(float), g[LABEL].to_numpy(float), method)
        if c is not None:
            ics.append(c)
    return float(np.mean(ics)) if ics else float("nan")


def run_random_kfold(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    alpha: float = 1e-2,
    seed: int = 0,
) -> ProtocolResult:
    """Shuffled K-fold over ROWS, ignoring time entirely.

    Shuffling rows rather than dates is the faithful reproduction of the mistake:
    it is what ``KFold(shuffle=True)`` does to a stacked panel, and it places
    observations of the same entity on adjacent dates into different folds.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    folds = np.array_split(idx, n_splits)

    out = []
    for k in range(n_splits):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        pred = _fit_predict(train, test, feature_cols, alpha)
        if pred is None:
            continue
        rows = test[[ENTITY, DATE, LABEL]].copy()
        rows[PRED] = pred
        out.append(rows)

    if not out:
        return ProtocolResult("random_kfold", "shuffled K-fold over rows", float("nan"), 0, 0)

    scored = pd.concat(out, ignore_index=True)
    return ProtocolResult(
        name="random_kfold",
        description=f"shuffled {n_splits}-fold over rows, time ignored",
        ic=_score_predictions(scored),
        n_folds=n_splits,
        n_test_rows=len(scored),
        detail={"seed": seed},
    )


def run_walk_forward(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    alpha: float = 1e-2,
    embargo: int = 0,
) -> ProtocolResult:
    """Expanding-window walk-forward, optionally with an embargo gap.

    Dates are split into ``n_splits + 1`` ordered blocks. Each fold trains on
    everything up to the block boundary minus the embargo, and tests on the next
    block. Training always precedes testing in time, and the embargo removes the
    dates immediately before the boundary -- the ones most correlated with the
    start of the test period.
    """
    dates = pd.DatetimeIndex(sorted(data[DATE].unique()))
    if len(dates) < (n_splits + 1) * 2:
        return ProtocolResult(
            "walk_forward", "expanding window", float("nan"), 0, 0,
            detail={"error": "not enough dates for the requested number of splits"},
        )

    blocks = np.array_split(np.arange(len(dates)), n_splits + 1)
    out = []
    for k in range(1, n_splits + 1):
        test_dates = set(dates[blocks[k]])
        boundary = blocks[k][0]
        # The embargo drops the last `embargo` training dates, not the first test
        # dates: shrinking the test set instead would change what is being scored
        # between protocols and confound the comparison.
        train_end_idx = max(0, boundary - embargo)
        train_dates = set(dates[:train_end_idx])
        if not train_dates or not test_dates:
            continue

        train = data.loc[data[DATE].isin(train_dates)]
        test = data.loc[data[DATE].isin(test_dates)]
        pred = _fit_predict(train, test, feature_cols, alpha)
        if pred is None:
            continue
        rows = test[[ENTITY, DATE, LABEL]].copy()
        rows[PRED] = pred
        out.append(rows)

    if not out:
        return ProtocolResult("walk_forward", "expanding window", float("nan"), 0, 0)

    scored = pd.concat(out, ignore_index=True)
    name = "purged_walk_forward" if embargo > 0 else "walk_forward"
    desc = (
        f"expanding window, {embargo}-date embargo"
        if embargo > 0
        else "expanding window, train strictly before test"
    )
    return ProtocolResult(
        name=name,
        description=desc,
        ic=_score_predictions(scored),
        n_folds=n_splits,
        n_test_rows=len(scored),
        detail={"embargo": embargo},
    )


def compare_protocols(
    panel: Panel,
    feature_cols: list[str] | None = None,
    n_splits: int = 5,
    embargo: int = 5,
    alpha: float = 1e-2,
    seed: int = 0,
) -> ProtocolComparison:
    """Score the same model under random, ordered and purged protocols.

    ``feature_cols`` defaults to columns prefixed ``f_``. The panel must carry
    features, not just predictions: the whole point is to refit under different
    splits, which a panel of finished predictions cannot support.
    """
    if feature_cols is None:
        feature_cols = [c for c in panel.data.columns if c.startswith("f_")]
    if not feature_cols:
        raise ValueError(
            "no feature columns found. This audit refits the model under each "
            "protocol, so it needs features (columns prefixed 'f_'), not only "
            "the finished predictions."
        )

    data = panel.data.dropna(subset=feature_cols + [LABEL]).copy()
    data = data.sort_values([DATE, ENTITY], kind="mergesort").reset_index(drop=True)

    results = [
        run_random_kfold(data, feature_cols, n_splits=n_splits, alpha=alpha, seed=seed),
        run_walk_forward(data, feature_cols, n_splits=n_splits, alpha=alpha, embargo=0),
        run_walk_forward(data, feature_cols, n_splits=n_splits, alpha=alpha, embargo=embargo),
    ]

    comp = ProtocolComparison(results=results, passed=None, verdict="")
    inflation = comp.inflation

    rnd = comp.by_name("random_kfold")
    purged = comp.by_name("purged_walk_forward")

    if not np.isfinite(inflation):
        comp.passed = None
        comp.verdict = "INCONCLUSIVE: one of the protocols could not be scored."
    elif inflation > MATERIAL_GAP:
        comp.passed = False
        comp.verdict = (
            f"FAIL: random K-fold scores {rnd.ic:+.4f} against "
            f"{purged.ic:+.4f} under a purged walk-forward -- an inflation of "
            f"{inflation:+.4f}. Random splitting places adjacent dates on both "
            "sides of the fold boundary, and for a persistent target those are "
            "near-duplicates, so the model is effectively evaluated on data it "
            "trained on. Only the walk-forward figure describes out-of-sample "
            "performance."
        )
    elif inflation < -MATERIAL_GAP:
        comp.passed = True
        comp.verdict = (
            f"PASS (unexpected direction): random K-fold scores LOWER by "
            f"{-inflation:.4f}. Worth checking the target is as persistent as "
            "assumed -- with little serial dependence, random splitting has "
            "little to leak, and the smaller training set in each fold can "
            "dominate."
        )
    else:
        comp.passed = True
        comp.verdict = (
            f"PASS: the protocols agree to within {abs(inflation):.4f}. The "
            "target carries little enough serial dependence that random "
            "splitting does not confer an advantage here."
        )

    comp.detail = {
        "n_splits": n_splits,
        "embargo": embargo,
        "n_feature_columns": len(feature_cols),
        "material_gap_threshold": MATERIAL_GAP,
    }
    return comp
