"""Panel data contract.

Every audit in this framework operates on a *panel*: multiple entities observed
over multiple dates. This module defines the single time convention the whole
codebase obeys, because ambiguity about "which day is t" is itself one of the
most common sources of the errors this tool is built to detect.

THE TIME CONVENTION
-------------------
A panel row is ``(entity_id, event_date, prediction, label)`` where:

* ``label`` is the *realized* value of the target ON ``event_date``.
* ``prediction`` is the forecast OF THAT SAME value, and it must have been
  computable using only information available STRICTLY BEFORE ``event_date``.

So both columns are indexed by the date the target is realized, not by the date
the forecast was made. Every row therefore answers one question: "on this day,
for this entity, what did we say would happen and what actually happened?"

Rationale for this choice (over indexing by forecast date):

1. It makes the correctness condition local and checkable. A row is valid iff
   its prediction used no information dated ``>= event_date``. There is no need
   to reason about a second date column, and no opportunity for an off-by-one
   between "the day the features come from" and "the day the label comes from".
2. Cross-sectional metrics group by ``event_date`` and are immediately
   meaningful: all labels in a group are realized simultaneously, so a
   cross-sectional correlation compares like with like.
3. Baselines become unambiguous. The persistence baseline is "the label at this
   entity's previous observed date", which is information available before
   ``event_date`` by construction.

Callers whose data is naturally keyed by forecast date must shift it before
building a ``Panel``. Doing that shift at the boundary, once, is safer than
letting two conventions coexist inside the library.

TRAIN / TEST BOUNDARY
---------------------
A ``Panel`` optionally carries ``train_end``: the last ``event_date`` belonging
to the training period. Several metrics (notably demeaned IC) require per-entity
statistics computed from training data ONLY -- using full-sample statistics to
demean is itself a form of leakage, and would quietly inflate the very number
this framework exists to deflate. Metrics that need it will raise rather than
silently fall back to the full sample.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

ENTITY = "entity_id"
DATE = "event_date"
PRED = "prediction"
LABEL = "label"

REQUIRED_COLUMNS = (ENTITY, DATE, PRED, LABEL)


class PanelError(ValueError):
    """Raised when input data violates the panel contract."""


@dataclass(frozen=True)
class Panel:
    """Validated panel of predictions and realized labels.

    Attributes
    ----------
    data:
        Long-format frame with columns ``entity_id, event_date, prediction,
        label``. Sorted by ``(entity_id, event_date)``. Additional columns are
        preserved -- group decomposition uses them for sector/exchange keys.
    train_end:
        Last ``event_date`` in the training period, or ``None`` if the panel
        represents a pure out-of-sample evaluation with no in-sample portion.
        Metrics requiring training-only statistics will raise if this is
        ``None``.
    label_name:
        Free-text description of the target, carried into reports so a report
        can never be misread as describing a different target.
    """

    data: pd.DataFrame
    train_end: pd.Timestamp | None = None
    label_name: str = "label"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        train_end: str | pd.Timestamp | None = None,
        label_name: str = "label",
        *,
        drop_incomplete: bool = True,
    ) -> Panel:
        """Validate and normalise a raw frame into a ``Panel``.

        Parameters
        ----------
        drop_incomplete:
            If True, rows where prediction or label is missing/non-finite are
            dropped (they cannot contribute to any correlation). If False, such
            rows raise. Dropping is the default because real panels routinely
            have gaps, but the count of dropped rows is reported so that silent
            attrition is visible.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise PanelError(
                f"missing required column(s): {missing}. "
                f"required = {list(REQUIRED_COLUMNS)}"
            )

        out = df.copy()
        out[DATE] = pd.to_datetime(out[DATE])
        for col in (PRED, LABEL):
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # Duplicate (entity, date) keys make cross-sectional metrics ill-defined
        # -- the same entity would enter one correlation twice with different
        # values. This is always a data-construction bug, so it raises.
        dupes = out.duplicated(subset=[ENTITY, DATE]).sum()
        if dupes:
            raise PanelError(
                f"{dupes} duplicate (entity_id, event_date) row(s); "
                "each entity may appear at most once per date"
            )

        finite = out[PRED].notna() & out[LABEL].notna()
        n_dropped = int((~finite).sum())
        if n_dropped and not drop_incomplete:
            raise PanelError(
                f"{n_dropped} row(s) with missing prediction or label; "
                "pass drop_incomplete=True to drop them"
            )
        out = out.loc[finite].copy()

        if out.empty:
            raise PanelError("panel is empty after dropping incomplete rows")

        out = out.sort_values([ENTITY, DATE], kind="mergesort").reset_index(drop=True)

        te = pd.to_datetime(train_end) if train_end is not None else None
        panel = cls(data=out, train_end=te, label_name=label_name)
        object.__setattr__(panel, "_n_dropped", n_dropped)
        return panel

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.data)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Sorted unique event dates present in the panel."""
        return pd.DatetimeIndex(sorted(self.data[DATE].unique()))

    @property
    def entities(self) -> list:
        return sorted(self.data[ENTITY].unique())

    @property
    def n_dropped(self) -> int:
        """Rows discarded during construction for missing values."""
        return getattr(self, "_n_dropped", 0)

    # ------------------------------------------------------------------
    # Train / test views
    # ------------------------------------------------------------------
    def require_train_end(self, metric_name: str) -> pd.Timestamp:
        """Return ``train_end`` or raise explaining why the metric needs it.

        Metrics call this instead of defaulting to the full sample. Falling back
        silently would compute a statistic on data the model was allowed to see,
        which is precisely the failure mode this framework detects elsewhere --
        it would be incoherent for the tool to commit it internally.
        """
        if self.train_end is None:
            raise PanelError(
                f"{metric_name} requires per-entity statistics computed from "
                "the training period only, but train_end is None. Supply "
                "train_end when building the Panel. (Using full-sample "
                "statistics here would leak out-of-sample information into the "
                "metric.)"
            )
        return self.train_end

    def train_slice(self) -> pd.DataFrame:
        """Rows with ``event_date <= train_end``."""
        te = self.require_train_end("train_slice")
        return self.data.loc[self.data[DATE] <= te]

    def test_slice(self) -> pd.DataFrame:
        """Rows with ``event_date > train_end``.

        Note this is the *evaluation* period. Cross-sectional metrics are
        normally computed here; computing them on the training period measures
        fit, not forecast skill.
        """
        te = self.require_train_end("test_slice")
        return self.data.loc[self.data[DATE] > te]

    def evaluation_view(self, scope: str = "test") -> pd.DataFrame:
        """Frame to compute metrics over.

        ``scope`` is one of ``test`` (default), ``train``, or ``all``. Making
        the scope explicit at every call site prevents the common mistake of
        reporting an in-sample number as though it were out-of-sample.
        """
        if scope == "test":
            return self.test_slice()
        if scope == "train":
            return self.train_slice()
        if scope == "all":
            return self.data
        raise ValueError(f"scope must be 'test', 'train' or 'all'; got {scope!r}")

    # ------------------------------------------------------------------
    # Helpers used by baselines and metrics
    # ------------------------------------------------------------------
    def with_column(self, name: str, values: pd.Series) -> Panel:
        """Return a new Panel with an extra column aligned on the panel index."""
        out = self.data.copy()
        out[name] = values.reindex(out.index)
        return Panel(data=out, train_end=self.train_end, label_name=self.label_name)

    def replace_prediction(self, values: pd.Series, *, label_name: str | None = None) -> Panel:
        """Return a new Panel whose ``prediction`` column is ``values``.

        Used to evaluate baselines through exactly the same code path as the
        model. Routing baselines through the identical metric implementation is
        deliberate: if the metric code had a bug, a separately-written baseline
        path could mask it, whereas a shared path makes the comparison
        apples-to-apples by construction.
        """
        out = self.data.copy()
        out[PRED] = values.reindex(out.index)
        keep = out[PRED].notna()
        out = out.loc[keep]
        if out.empty:
            raise PanelError("replacement prediction is empty after alignment")
        return Panel(
            data=out.reset_index(drop=True),
            train_end=self.train_end,
            label_name=label_name or self.label_name,
        )

    def per_entity_train_mean(self, column: str) -> pd.Series:
        """Mean of ``column`` per entity over the TRAINING period only.

        Returned as a Series indexed by ``entity_id``. Entities with no training
        observations get ``NaN`` -- callers must decide whether to drop them
        rather than have a fabricated value substituted here.
        """
        self.require_train_end("per_entity_train_mean")
        tr = self.train_slice()
        return tr.groupby(ENTITY)[column].mean()

    def cross_sections(self, scope: str = "test") -> Iterable[tuple[pd.Timestamp, pd.DataFrame]]:
        """Yield ``(date, frame)`` for each event date in the chosen scope."""
        view = self.evaluation_view(scope)
        yield from view.groupby(DATE, sort=True)

    def describe(self) -> dict:
        """Summary used in report headers so every report states its own scope."""
        return {
            "n_rows": self.n_rows,
            "n_entities": len(self.entities),
            "n_dates": len(self.dates),
            "first_date": str(self.dates[0].date()),
            "last_date": str(self.dates[-1].date()),
            "train_end": str(self.train_end.date()) if self.train_end is not None else None,
            "label_name": self.label_name,
            "rows_dropped_incomplete": self.n_dropped,
        }
