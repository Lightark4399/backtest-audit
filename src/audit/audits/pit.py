"""Point-in-time audit: was the data knowable when the forecast claims to have been made?

The other modules ask whether a score is *earned*. This one asks whether it was
*possible*.

A backtest run against a mutable table reads the current best value for every
past date. If any of those values were corrected after the fact, the pipeline has
consumed information that did not exist at the time -- and nothing in the data
reveals it, because the corrected value looks exactly like an original one. This
is the failure mode the alignment audit explicitly cannot detect: the prediction
really does match the label it is scored against, and the pairing is sound. What
is unsound is the vintage.

The measurement
---------------
Run the same feature construction twice over the same history:

* **restated** -- the current best value for every date, corrections included.
* **as-of** -- for each date, the value as it actually stood on that date.

Then score the same model both ways. The gap is the look-ahead advantage,
expressed in the same units as everything else in the report.

Two design points worth stating, because each removes an alternative explanation
for the gap:

**Both vintages go through the identical SQL macro.** If features were computed
two different ways, an IC difference could be an artefact of the feature code
rather than of the data. ``features_from`` is applied to both relations.

**The as-of pass uses a rolling cutoff, not a single snapshot.** Taking one
snapshot at the start of the evaluation period would understate what a
practitioner actually had: by mid-period they would have received corrections for
early dates. The honest reconstruction gives each date the data available on that
date, which is what ``observation_asof`` returns, evaluated per date.

When the gap will be zero, and why that is not a clean bill of health
---------------------------------------------------------------------
If nothing was ever revised, the two vintages coincide and this audit reports
nothing. That is a true statement about this channel of look-ahead, not about
look-ahead in general: a feature that uses same-day data is unknowable in time
even when no value was ever corrected. The report says so rather than letting a
zero read as an all-clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..ingest.duckdb_store import BitemporalStore, RevisionSpec
from ..metrics.ic import cross_sectional_ic, demeaned_ic
from ..panel import DATE, ENTITY, LABEL, PRED, Panel

# Below this the difference between the two vintages is indistinguishable from
# estimation noise on a panel of this size.
MATERIAL_GAP = 0.01


@dataclass
class PITResult:
    """Outcome of comparing a restated backtest against a point-in-time one."""

    restated_ic: float
    asof_ic: float
    restated_demeaned_ic: float
    asof_demeaned_ic: float
    n_revisions: int
    n_observations: int
    revision_rate: float
    mean_revision_size: float
    mean_revision_lag_days: float
    passed: bool | None
    verdict: str
    detail: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        """Look-ahead advantage on the demeaned IC, the headline metric."""
        return self.restated_demeaned_ic - self.asof_demeaned_ic

    @property
    def gap_ratio(self) -> float:
        if not np.isfinite(self.asof_demeaned_ic) or abs(self.asof_demeaned_ic) < 1e-12:
            return float("nan")
        return self.gap / abs(self.asof_demeaned_ic)

    def to_dict(self) -> dict:
        return {
            "restated_ic": self.restated_ic,
            "asof_ic": self.asof_ic,
            "restated_demeaned_ic": self.restated_demeaned_ic,
            "asof_demeaned_ic": self.asof_demeaned_ic,
            "gap": self.gap,
            "gap_ratio": self.gap_ratio,
            "n_revisions": self.n_revisions,
            "n_observations": self.n_observations,
            "revision_rate": self.revision_rate,
            "mean_revision_size": self.mean_revision_size,
            "mean_revision_lag_days": self.mean_revision_lag_days,
            "passed": self.passed,
            "verdict": self.verdict,
            **self.detail,
        }


def _fit_predict_per_entity(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    train_end: pd.Timestamp,
    feature_cols: list[str],
    alpha: float = 1e-2,
    min_train: int = 30,
) -> pd.DataFrame:
    """Per-entity ridge, fit on training rows, predicting the evaluation period.

    Deliberately the same simple estimator for both vintages, with a fixed alpha.
    Tuning would introduce a second source of difference between the two runs and
    make the gap harder to attribute -- the question here is about the data, not
    about the model.
    """
    df = features.merge(labels, on=[ENTITY, "event_date"], how="inner", suffixes=("", "_lab"))
    df = df.rename(columns={"value": LABEL})
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.dropna(subset=feature_cols + [LABEL])

    out = []
    for _entity, g in df.groupby(ENTITY, sort=False):
        g = g.sort_values("event_date", kind="mergesort")
        tr = g.loc[g["event_date"] <= train_end]
        te = g.loc[g["event_date"] > train_end]
        if len(tr) < min_train or te.empty:
            continue

        xtr = np.column_stack([np.ones(len(tr)), tr[feature_cols].to_numpy(float)])
        ytr = tr[LABEL].to_numpy(float)
        reg = np.eye(xtr.shape[1]) * alpha
        reg[0, 0] = 0.0
        try:
            w = np.linalg.solve(xtr.T @ xtr + reg, xtr.T @ ytr)
        except np.linalg.LinAlgError:
            continue

        xte = np.column_stack([np.ones(len(te)), te[feature_cols].to_numpy(float)])
        pred = te[[ENTITY, "event_date", LABEL]].copy()
        pred[PRED] = xte @ w
        # Training rows are carried through so the audit can compute per-entity
        # training means; they get the training-mean label as a placeholder
        # prediction, which never enters an evaluation-period metric.
        tr_rows = tr[[ENTITY, "event_date", LABEL]].copy()
        tr_rows[PRED] = tr[LABEL].mean()
        out.append(pd.concat([tr_rows, pred], ignore_index=True))

    if not out:
        raise RuntimeError("no entity produced predictions")

    res = pd.concat(out, ignore_index=True).rename(columns={"event_date": DATE})
    return res[[ENTITY, DATE, PRED, LABEL]]


def build_asof_features(
    store: BitemporalStore,
    dates: pd.DatetimeIndex,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Features where each date is computed from the data available on that date.

    A single snapshot would misstate the exercise in one direction or the other:
    taken early it denies the practitioner corrections they would have received,
    taken late it hands them corrections they had not yet seen. Rebuilding per
    date is the faithful reconstruction.

    Cost is one pass over the history per date, which is the price of the
    reconstruction being honest. For long histories a caller can thin ``dates``;
    the audit does so by default outside the evaluation window, where the exact
    vintage of a training feature matters far less than in the period being
    scored.
    """
    frames = []
    for d in dates:
        snap = store.features("asof", asof=d)
        if snap.empty:
            continue
        snap["event_date"] = pd.to_datetime(snap["event_date"])
        row = snap.loc[snap["event_date"] == d]
        if not row.empty:
            frames.append(row)
    if not frames:
        raise RuntimeError("as-of reconstruction produced no rows")
    return pd.concat(frames, ignore_index=True)


def run_pit_audit(
    observations: pd.DataFrame,
    labels: pd.DataFrame,
    train_end,
    revisions: RevisionSpec | None = None,
    feature_cols: list[str] | None = None,
    max_asof_dates: int = 60,
) -> PITResult:
    """Compare a restated backtest against a point-in-time one.

    Parameters
    ----------
    observations, labels:
        Long frames with ``entity_id, event_date, value``. ``observations`` holds
        the TRUE values; the revision history is constructed from them.
    revisions:
        How the revision scenario is built. ``None`` means no revisions, in which
        case the two vintages coincide by construction -- useful as a control
        that the comparison reports zero when there is nothing to find.
    max_asof_dates:
        Cap on the number of evaluation dates reconstructed as-of. Each one costs
        a pass over the history, so the audit samples evenly across the
        evaluation window rather than running unbounded.
    """
    feature_cols = feature_cols or ["f_lag1", "f_mean5", "f_mean20", "f_sd20"]
    train_end = pd.Timestamp(train_end)

    store = BitemporalStore()
    load_stats = store.load_observations(observations, revisions=revisions)
    store.load_labels(labels)

    rev = store.revision_summary()
    n_obs = len(store.restated())

    # ---- restated vintage: what a naive backtest sees ----
    f_restated = store.features("restated")
    f_restated["event_date"] = pd.to_datetime(f_restated["event_date"])
    lab = store.labels()
    lab["event_date"] = pd.to_datetime(lab["event_date"])

    # ---- as-of vintage: what was actually available ----
    all_dates = pd.DatetimeIndex(sorted(f_restated["event_date"].unique()))
    eval_dates = all_dates[all_dates > train_end]
    if len(eval_dates) > max_asof_dates:
        idx = np.linspace(0, len(eval_dates) - 1, max_asof_dates).astype(int)
        eval_dates = eval_dates[idx]

    # Training-period features come from the restated view in both arms. This is
    # deliberate and conservative: it isolates the gap to the evaluation period,
    # so the number reported is attributable to scoring on data nobody had rather
    # than to the two models having been fit on different inputs.
    f_asof_eval = build_asof_features(store, eval_dates, feature_cols)
    f_train = f_restated.loc[f_restated["event_date"] <= train_end]
    f_asof = pd.concat([f_train, f_asof_eval], ignore_index=True)

    # BOTH arms must be scored on exactly the same evaluation dates. The as-of
    # reconstruction is sampled (each date costs a pass over the history), so
    # leaving the restated arm on the full set would make the two panels differ in
    # composition as well as in vintage -- and the gap would then be partly a
    # comparison of different samples. Measured on the no-revision control, that
    # confound alone produced a gap of -0.013, larger than the threshold the audit
    # uses to call a result material. Restricting to the shared dates removes it:
    # with no revisions the two panels become identical and the gap is exactly 0.
    f_restated_scored = pd.concat(
        [f_train, f_restated.loc[f_restated["event_date"].isin(eval_dates)]],
        ignore_index=True,
    )

    panel_restated = Panel.from_frame(
        _fit_predict_per_entity(f_restated_scored, lab, train_end, feature_cols),
        train_end=train_end,
        label_name="pit_restated",
    )
    panel_asof = Panel.from_frame(
        _fit_predict_per_entity(f_asof, lab, train_end, feature_cols),
        train_end=train_end,
        label_name="pit_asof",
    )

    restated_ic = cross_sectional_ic(panel_restated, method="spearman").mean
    asof_ic = cross_sectional_ic(panel_asof, method="spearman").mean
    restated_dm = demeaned_ic(panel_restated).mean
    asof_dm = demeaned_ic(panel_asof).mean

    gap = restated_dm - asof_dm
    n_rev = len(rev)
    rate = n_rev / n_obs if n_obs else float("nan")

    if n_rev == 0:
        passed = None
        verdict = (
            "NO REVISIONS: the restated and point-in-time views are identical, so "
            "this channel of look-ahead is absent. Note this is not a clean bill "
            "of health -- a feature built from same-day data is unknowable in "
            "time whether or not any value was ever corrected."
        )
    elif np.isfinite(gap) and gap > MATERIAL_GAP:
        passed = False
        verdict = (
            f"FAIL: the restated backtest scores {gap:+.4f} higher than the "
            f"point-in-time one ({restated_dm:+.4f} vs {asof_dm:+.4f}). "
            f"{n_rev:,} of {n_obs:,} observations ({rate:.1%}) were corrected "
            "after the fact, and the backtest consumed those corrections. That "
            "advantage was not available at the time and will not be available "
            "in production."
        )
    elif np.isfinite(gap) and gap < -MATERIAL_GAP:
        passed = True
        verdict = (
            f"PASS (unexpected direction): the point-in-time backtest scores "
            f"HIGHER by {-gap:+.4f}. Corrections made the data less predictive "
            "here, so the restated run understates rather than flatters. Worth "
            "checking the revision construction is doing what was intended."
        )
    else:
        passed = True
        verdict = (
            f"PASS: restated and point-in-time scores agree to within "
            f"{abs(gap):.4f} despite {n_rev:,} corrections ({rate:.1%}). The "
            "revisions carried no information about the target, so using them "
            "conferred no advantage."
        )

    result = PITResult(
        restated_ic=restated_ic,
        asof_ic=asof_ic,
        restated_demeaned_ic=restated_dm,
        asof_demeaned_ic=asof_dm,
        n_revisions=n_rev,
        n_observations=n_obs,
        revision_rate=rate,
        mean_revision_size=float(rev["revision_size"].abs().mean()) if n_rev else 0.0,
        mean_revision_lag_days=float(rev["revision_lag_days"].mean()) if n_rev else 0.0,
        passed=passed,
        verdict=verdict,
        detail={
            "n_versions_stored": load_stats.get("n_versions"),
            "asof_dates_reconstructed": len(eval_dates),
            "material_gap_threshold": MATERIAL_GAP,
        },
    )
    store.close()
    return result
