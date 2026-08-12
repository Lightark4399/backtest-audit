-- Schema for the backtest credibility audit framework.
--
-- The organising idea is that every observation carries TWO time dimensions:
--
--   event_date    the business date the data describes
--   knowledge_date the date on which that value first became available
--
-- Keeping both is what makes look-ahead bias detectable. With only event_date,
-- a corrected value silently overwrites the original and the historical record
-- becomes the *current* view of the past rather than what was actually knowable
-- at the time. Any backtest run against that record can consume information
-- from the future without anything in the data revealing it.
--
-- Storing revisions as additional rows (rather than UPDATEing in place) means
-- the table is append-only for corrections, and a point-in-time query is a
-- filter on knowledge_date. See 002_pit_views.sql for the as-of join.

BEGIN;

-- ---------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities (
    entity_id     SERIAL PRIMARY KEY,
    ticker        TEXT        NOT NULL UNIQUE,
    name          TEXT,
    exchange      TEXT,
    sector        TEXT,
    -- Listing and delisting dates exist so that a historical universe can be
    -- reconstructed WITHOUT survivorship bias. Selecting entities that exist
    -- today and backfilling their history is one of the most common ways a
    -- backtest is quietly inflated: the failures have been removed from the
    -- sample. A universe query must filter on these dates instead.
    listing_date   DATE,
    delisting_date DATE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT entities_dates_ordered
        CHECK (delisting_date IS NULL OR listing_date IS NULL
               OR delisting_date >= listing_date)
);

COMMENT ON COLUMN entities.delisting_date IS
    'NULL means still listed. Required for survivorship-bias-free universes.';

-- ---------------------------------------------------------------------------
-- Raw market data, bitemporal
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS price_raw (
    price_id       BIGSERIAL PRIMARY KEY,
    entity_id      INTEGER     NOT NULL REFERENCES entities(entity_id),
    event_date     DATE        NOT NULL,
    -- When this particular version of the row became known. The original
    -- observation normally has knowledge_date = event_date (or the next
    -- morning); a later correction is inserted as a NEW row with a LATER
    -- knowledge_date, leaving the original intact.
    knowledge_date DATE        NOT NULL,
    open           NUMERIC(18, 6),
    high           NUMERIC(18, 6),
    low            NUMERIC(18, 6),
    close          NUMERIC(18, 6),
    volume         NUMERIC(20, 4),
    source         TEXT        NOT NULL DEFAULT 'unknown',
    revision_note  TEXT,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- One version per (entity, event_date, knowledge_date). Two different values
    -- claiming to have been known on the same day for the same date is a
    -- contradiction, not a revision, so it is rejected here rather than being
    -- resolved arbitrarily at query time.
    CONSTRAINT price_raw_version_unique
        UNIQUE (entity_id, event_date, knowledge_date),

    -- A value cannot be known before the day it describes.
    CONSTRAINT price_raw_knowledge_not_before_event
        CHECK (knowledge_date >= event_date),

    -- Basic OHLC coherence. Rejecting incoherent bars at write time keeps the
    -- distinction clear between "the data is odd" (a modelling question) and
    -- "the data is impossible" (an ingestion bug).
    CONSTRAINT price_raw_ohlc_coherent
        CHECK (
            (open IS NULL OR open > 0) AND
            (high IS NULL OR high > 0) AND
            (low  IS NULL OR low  > 0) AND
            (close IS NULL OR close > 0) AND
            (high IS NULL OR low IS NULL OR high >= low) AND
            (high IS NULL OR open IS NULL OR high >= open) AND
            (high IS NULL OR close IS NULL OR high >= close) AND
            (low IS NULL OR open IS NULL OR low <= open) AND
            (low IS NULL OR close IS NULL OR low <= close)
        )
);

-- The workhorse index: PIT queries filter by entity and event_date, then pick
-- the newest knowledge_date at or before the as-of date. DESC on
-- knowledge_date lets that pick be an index scan taking the first row.
CREATE INDEX IF NOT EXISTS price_raw_pit_idx
    ON price_raw (entity_id, event_date, knowledge_date DESC);

CREATE INDEX IF NOT EXISTS price_raw_event_date_idx
    ON price_raw (event_date);

-- ---------------------------------------------------------------------------
-- Derived data
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS features (
    entity_id    INTEGER NOT NULL REFERENCES entities(entity_id),
    event_date   DATE    NOT NULL,
    feature_name TEXT    NOT NULL,
    value        DOUBLE PRECISION,
    -- The as-of date of the price snapshot this feature was computed from.
    -- Without it, a feature table cannot be attributed to a data vintage and
    -- the PIT-vs-restated comparison in module 3 is impossible.
    computed_asof DATE   NOT NULL,
    PRIMARY KEY (entity_id, event_date, feature_name, computed_asof)
);

CREATE TABLE IF NOT EXISTS labels (
    entity_id     INTEGER NOT NULL REFERENCES entities(entity_id),
    event_date    DATE    NOT NULL,
    label_name    TEXT    NOT NULL,
    value         DOUBLE PRECISION,
    computed_asof DATE    NOT NULL,
    -- Why a label might not be usable even though a number exists: a limit-up
    -- day with zero intraday range, a suspension, a stub quote. Recording the
    -- reason (rather than deleting the row) keeps the exclusion auditable and
    -- lets the exclusion rate itself be monitored.
    excluded_reason TEXT,
    PRIMARY KEY (entity_id, event_date, label_name, computed_asof)
);

COMMENT ON COLUMN labels.excluded_reason IS
    'NULL means usable. Non-NULL rows are retained so exclusions stay auditable.';

CREATE TABLE IF NOT EXISTS predictions (
    run_id          TEXT    NOT NULL,
    entity_id       INTEGER NOT NULL REFERENCES entities(entity_id),
    -- Convention (mirrors src/audit/panel.py): event_date is the date the
    -- target is REALIZED. The prediction stored here must have been computable
    -- from information dated strictly before it.
    event_date      DATE    NOT NULL,
    predicted_value DOUBLE PRECISION,
    PRIMARY KEY (run_id, entity_id, event_date)
);

-- ---------------------------------------------------------------------------
-- Audit bookkeeping
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit_runs (
    run_id      TEXT PRIMARY KEY,
    config_hash TEXT        NOT NULL,
    config_json JSONB       NOT NULL,
    git_commit  TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status      TEXT        NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'ok', 'failed'))
);

CREATE TABLE IF NOT EXISTS audit_results (
    run_id          TEXT   NOT NULL REFERENCES audit_runs(run_id),
    metric_name     TEXT   NOT NULL,
    metric_value    DOUBLE PRECISION,
    diagnostic_json JSONB,
    PRIMARY KEY (run_id, metric_name)
);

COMMENT ON TABLE audit_results IS
    'metric_value is NULL when a metric was undefined; diagnostic_json carries '
    'the reason. NULL and 0.0 mean different things and are not interchangeable.';

COMMIT;
