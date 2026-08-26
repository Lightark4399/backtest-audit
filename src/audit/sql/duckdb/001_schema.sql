-- DuckDB dialect of the bitemporal schema.
--
-- The Postgres version in sql/001_schema.sql and sql/002_pit_views.sql is the
-- reference design. This file exists so the point-in-time logic can be executed
-- and tested for real -- DuckDB is embedded, so integration tests run in CI with
-- no service to provision, whereas a Postgres-backed test needs a container that
-- not every environment can supply.
--
-- The dialects agree on everything that matters here: DISTINCT ON, window frames,
-- CTEs and CHECK constraints all behave identically. Two differences worth
-- naming, because they are the kind of thing that silently breaks a port:
--
--   * Postgres set-returning functions (price_asof(date)) become table MACROs.
--     Same call syntax, same semantics, different DDL keyword.
--   * ASOF is a reserved word in DuckDB (it has a native ASOF JOIN), so the
--     macro parameter is named cutoff rather than asof. Worth noting because
--     the failure is a parse error several lines away from the actual clash.
--   * DuckDB has no SERIAL; identity columns come from a sequence, and since
--     nothing here depends on a generated key, natural keys are used instead.
--
-- Anything that reads from these objects is written to work against both.

-- ---------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities (
    entity_id      VARCHAR PRIMARY KEY,
    sector         VARCHAR,
    -- Listing and delisting dates exist so a historical universe can be
    -- reconstructed WITHOUT survivorship bias. Selecting the entities that exist
    -- today and backfilling their history removes exactly the failures that a
    -- backtest most needs to see.
    listing_date   DATE,
    delisting_date DATE,
    CHECK (delisting_date IS NULL OR listing_date IS NULL
           OR delisting_date >= listing_date)
);

-- ---------------------------------------------------------------------------
-- Bitemporal observations
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS observation_raw (
    entity_id      VARCHAR NOT NULL,
    -- The business date the observation describes.
    event_date     DATE    NOT NULL,
    -- When this VERSION of the value became available. The first report normally
    -- has knowledge_date = event_date; a later correction is inserted as a NEW
    -- row with a LATER knowledge_date, leaving the original intact.
    knowledge_date DATE    NOT NULL,
    value          DOUBLE,
    source         VARCHAR DEFAULT 'initial',
    revision_note  VARCHAR,

    -- One version per (entity, event_date, knowledge_date). Two different values
    -- claiming to have been known on the same day for the same date is a
    -- contradiction rather than a revision.
    PRIMARY KEY (entity_id, event_date, knowledge_date),

    -- A value cannot be known before the day it describes.
    CHECK (knowledge_date >= event_date)
);

CREATE TABLE IF NOT EXISTS labels_raw (
    entity_id  VARCHAR NOT NULL,
    event_date DATE    NOT NULL,
    value      DOUBLE,
    PRIMARY KEY (entity_id, event_date)
);

-- ---------------------------------------------------------------------------
-- The two views of history
-- ---------------------------------------------------------------------------

-- RESTATED: the current best value for each (entity, event_date), including
-- every correction made since. This is what a naive query against a mutable
-- table returns, and it must NOT be used to simulate a historical decision.
CREATE OR REPLACE VIEW observation_restated AS
SELECT DISTINCT ON (entity_id, event_date)
    entity_id,
    event_date,
    knowledge_date,
    value
FROM observation_raw
ORDER BY entity_id, event_date, knowledge_date DESC;

-- AS-OF: the data as it actually stood on a given date. A macro rather than a
-- view because the cutoff is a parameter.
--
-- The two filters do different jobs and both are required:
--   knowledge_date <= asof  -- the version had been published by then
--   event_date     <= asof  -- the bar itself had happened by then
-- Dropping the first is the classic error: it returns today's corrected values
-- for past dates, which looks like history and is not.
CREATE OR REPLACE MACRO observation_asof(cutoff) AS TABLE
    SELECT DISTINCT ON (entity_id, event_date)
        entity_id,
        event_date,
        knowledge_date,
        value
    FROM observation_raw
    WHERE knowledge_date <= cutoff
      AND event_date <= cutoff
    ORDER BY entity_id, event_date, knowledge_date DESC;

-- Universe membership decided by listing/delisting dates rather than by present
-- existence, so an entity that delisted mid-history stays in the sample for the
-- dates it actually traded.
CREATE OR REPLACE MACRO universe_asof(cutoff) AS TABLE
    SELECT entity_id, sector
    FROM entities
    WHERE (listing_date   IS NULL OR listing_date   <= cutoff)
      AND (delisting_date IS NULL OR delisting_date  > cutoff);

-- ---------------------------------------------------------------------------
-- Revision diagnostics
-- ---------------------------------------------------------------------------

-- Every (entity, event_date) that was ever corrected, with the first and final
-- values. This is the raw material for the point-in-time audit: if nothing was
-- ever revised, restated and as-of views coincide and no look-ahead is possible
-- through this channel.
CREATE OR REPLACE VIEW revisions AS
WITH first_version AS (
    SELECT DISTINCT ON (entity_id, event_date)
        entity_id, event_date, knowledge_date AS first_known, value AS first_value
    FROM observation_raw
    ORDER BY entity_id, event_date, knowledge_date ASC
),
last_version AS (
    SELECT DISTINCT ON (entity_id, event_date)
        entity_id, event_date, knowledge_date AS last_known, value AS last_value
    FROM observation_raw
    ORDER BY entity_id, event_date, knowledge_date DESC
)
SELECT
    f.entity_id,
    f.event_date,
    f.first_known,
    l.last_known,
    f.first_value,
    l.last_value,
    l.last_value - f.first_value AS revision_size,
    l.last_known - f.first_known AS revision_lag_days
FROM first_version f
JOIN last_version l USING (entity_id, event_date)
WHERE f.first_known <> l.last_known;

-- ---------------------------------------------------------------------------
-- Feature construction with strictly-past windows
-- ---------------------------------------------------------------------------
--
-- The frame clause is the point:
--
--     ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
--
-- SQL's DEFAULT frame includes the current row. A "trailing mean" written
-- without an explicit frame therefore contains same-day information while still
-- looking like a plausible moving average, which is how the error survives
-- review. Ending at 1 PRECEDING states the exclusion.
--
-- Written as a macro taking a table NAME so the identical feature logic can be
-- applied to the restated view and to any as-of snapshot. Computing features two
-- different ways for the two views would confound the comparison: any IC
-- difference could then be an artefact of the feature code rather than of the
-- data vintage.
--
-- query_table() is DuckDB's way of parameterising over a relation: a plain macro
-- parameter is an expression, not a table reference, so `FROM tbl` fails to
-- resolve. The argument is therefore a string naming the relation.
CREATE OR REPLACE MACRO features_from(tbl) AS TABLE
    SELECT
        entity_id,
        event_date,
        LAG(value, 1) OVER w AS f_lag1,
        AVG(value) OVER (
            PARTITION BY entity_id ORDER BY event_date
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS f_mean5,
        AVG(value) OVER (
            PARTITION BY entity_id ORDER BY event_date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS f_mean20,
        STDDEV_SAMP(value) OVER (
            PARTITION BY entity_id ORDER BY event_date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS f_sd20
    FROM query_table(tbl)
    WINDOW w AS (PARTITION BY entity_id ORDER BY event_date);
