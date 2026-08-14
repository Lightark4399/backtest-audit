"""Bitemporal store backed by DuckDB.

Holds observations with both a business date and a knowledge date, so the same
history can be read two ways: as it stands now, and as it actually stood on a
given day. The difference between those two readings is what the point-in-time
audit measures.

Why an embedded database rather than a dataframe
------------------------------------------------
The as-of join is the operation being demonstrated, and expressing it in SQL is
the point: it is how the problem is solved in the systems this work is about, and
it makes the correctness condition inspectable -- a reviewer can read the WHERE
clause and see that ``knowledge_date <= asof`` is enforced. Reimplementing it in
pandas would hide that behind index arithmetic.

DuckDB rather than Postgres is a deployment choice, not a design one: the schema
is written to run on both, and the Postgres version is the reference. Embedded
means the integration tests run anywhere, including CI, with nothing to
provision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "duckdb is required for the point-in-time module. "
        "Install it with: pip install duckdb"
    ) from exc

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "sql" / "duckdb" / "001_schema.sql"


@dataclass
class RevisionSpec:
    """How to construct a revision scenario.

    Revisions matter for backtesting only when the revised value carries
    information about the future. That is not automatic -- a correction that
    replaces one unbiased measurement with another equally unbiased one leaves
    the restated view no more predictive than the original.

    The mechanism modelled here is the one that does leak: an observation is
    initially reported with error and later corrected toward its true value. The
    restated series is therefore *closer to the truth* than anything that existed
    at the time, and a pipeline built on it has been handed information nobody
    had. This is realistic -- late-arriving corrections, restatements and
    re-derived adjustment factors all work this way.

    Attributes
    ----------
    fraction:
        Share of observations initially misreported.
    error_scale:
        Size of the initial error, as a multiple of the series' own standard
        deviation.
    lag_days:
        How long the correction takes to arrive. Longer lags mean a larger
        window during which a naive backtest is using data nobody had.
    seed:
        Fixed so a scenario is reproducible; an audit whose inputs change between
        runs cannot be checked by a reviewer.
    """

    fraction: float = 0.30
    error_scale: float = 1.0
    lag_days: int = 5
    seed: int = 17


class BitemporalStore:
    """DuckDB-backed store of observations, labels and entity metadata."""

    def __init__(self, path: str | None = None):
        # ':memory:' by default: these stores are built per audit run and are
        # not intended to persist. A path can be supplied to inspect one.
        self.con = duckdb.connect(path or ":memory:")
        self._apply_schema()

    def _apply_schema(self) -> None:
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"schema not found at {SCHEMA_PATH}")
        self.con.execute(SCHEMA_PATH.read_text())

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_observations(
        self,
        frame: pd.DataFrame,
        revisions: RevisionSpec | None = None,
    ) -> dict:
        """Insert observations, optionally with a constructed revision history.

        ``frame`` must have ``entity_id, event_date, value`` and holds the TRUE
        values. When ``revisions`` is given, a subset is first published with an
        error and corrected ``lag_days`` later; both versions are stored, so the
        table records what was believed at each point rather than only what turned
        out to be so.
        """
        df = frame[["entity_id", "event_date", "value"]].copy()
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.date

        if revisions is None:
            df["knowledge_date"] = df["event_date"]
            df["source"] = "initial"
            df["revision_note"] = None
            rows = df
            stats = {"n_revised": 0, "n_versions": len(df)}
        else:
            rng = np.random.default_rng(revisions.seed)
            n = len(df)
            revised_mask = rng.random(n) < revisions.fraction
            sd = float(np.nanstd(df["value"].to_numpy(dtype=float))) or 1.0
            error = rng.normal(0.0, revisions.error_scale * sd, size=n)

            # Version 1: what was published at the time. Correct for most rows,
            # wrong for the revised subset.
            v1 = df.copy()
            v1["value"] = np.where(revised_mask, df["value"] + error, df["value"])
            v1["knowledge_date"] = v1["event_date"]
            v1["source"] = "initial"
            v1["revision_note"] = None

            # Version 2: the correction, published lag_days later, only for rows
            # that were wrong. Inserted as a NEW row -- never an update, or the
            # original belief would be lost and the history would become
            # unfalsifiable.
            v2 = df.loc[revised_mask].copy()
            v2["knowledge_date"] = pd.to_datetime(v2["event_date"]) + pd.Timedelta(
                days=revisions.lag_days
            )
            v2["knowledge_date"] = v2["knowledge_date"].dt.date
            v2["source"] = "revision"
            v2["revision_note"] = "corrected to final value"

            rows = pd.concat([v1, v2], ignore_index=True)
            stats = {"n_revised": int(revised_mask.sum()), "n_versions": len(rows)}

        rows = rows[
            ["entity_id", "event_date", "knowledge_date", "value", "source", "revision_note"]
        ]
        self.con.register("_incoming", rows)
        self.con.execute("INSERT INTO observation_raw SELECT * FROM _incoming")
        self.con.unregister("_incoming")
        return stats

    def load_labels(self, frame: pd.DataFrame) -> None:
        """Insert labels. Labels are not versioned here.

        Deliberate scope limit: the audit asks whether *features* were knowable
        when the forecast was made. Label revisions are a real phenomenon but a
        separate question, and conflating them would make the measured look-ahead
        impossible to attribute.
        """
        df = frame[["entity_id", "event_date", "value"]].copy()
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
        self.con.register("_labels", df)
        self.con.execute("INSERT INTO labels_raw SELECT * FROM _labels")
        self.con.unregister("_labels")

    def load_entities(self, frame: pd.DataFrame) -> None:
        """Insert entity metadata including listing and delisting dates."""
        cols = ["entity_id", "sector", "listing_date", "delisting_date"]
        df = frame.reindex(columns=cols).copy()
        for c in ("listing_date", "delisting_date"):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
        self.con.register("_entities", df)
        self.con.execute("INSERT INTO entities SELECT * FROM _entities")
        self.con.unregister("_entities")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def restated(self) -> pd.DataFrame:
        """Current best value per (entity, event_date) -- corrections included."""
        return self.con.execute(
            "SELECT entity_id, event_date, value FROM observation_restated "
            "ORDER BY entity_id, event_date"
        ).df()

    def asof(self, date) -> pd.DataFrame:
        """Data as actually known on ``date``."""
        d = pd.Timestamp(date).date()
        return self.con.execute(
            "SELECT entity_id, event_date, value FROM observation_asof(?) "
            "ORDER BY entity_id, event_date",
            [d],
        ).df()

    def features(self, view: str = "restated", asof=None) -> pd.DataFrame:
        """Lagged features computed through the shared SQL macro.

        Both views go through ``features_from``, so an IC difference between them
        cannot be an artefact of two different feature implementations.
        """
        if view == "restated":
            return self.con.execute(
                "SELECT * FROM features_from('observation_restated') "
                "ORDER BY entity_id, event_date"
            ).df()
        if view == "asof":
            if asof is None:
                raise ValueError("view='asof' requires an asof date")
            d = pd.Timestamp(asof).date()
            # DuckDB cannot bind a prepared parameter inside CREATE VIEW, so the
            # date is inlined. It is formatted from a parsed Timestamp rather
            # than interpolated from caller input, so the literal is always a
            # well-formed ISO date and cannot carry anything else.
            self.con.execute(
                "CREATE OR REPLACE TEMP VIEW _snapshot AS "
                f"SELECT * FROM observation_asof(DATE '{d.isoformat()}')"
            )
            return self.con.execute(
                "SELECT * FROM features_from('_snapshot') ORDER BY entity_id, event_date"
            ).df()
        raise ValueError(f"view must be 'restated' or 'asof'; got {view!r}")

    def revision_summary(self) -> pd.DataFrame:
        """Every observation that was ever corrected, with size and lag."""
        return self.con.execute("SELECT * FROM revisions ORDER BY event_date").df()

    def universe(self, date) -> list[str]:
        """Entities that were listed and not yet delisted on ``date``."""
        d = pd.Timestamp(date).date()
        out = self.con.execute("SELECT entity_id FROM universe_asof(?)", [d]).df()
        return sorted(out["entity_id"].tolist())

    def labels(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT entity_id, event_date, value FROM labels_raw ORDER BY entity_id, event_date"
        ).df()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> BitemporalStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
