-- DuckDB dialect of the bitemporal schema.
--
-- This runtime copy lives inside the Python package so wheels and sdists do not
-- depend on the repository-level sql/ directory. The Postgres schema in that
-- directory remains the reference design.

CREATE TABLE IF NOT EXISTS entities (
    entity_id      VARCHAR PRIMARY KEY,
    sector         VARCHAR,
    listing_date   DATE,
    delisting_date DATE,
    CHECK (delisting_date IS NULL OR listing_date IS NULL
           OR delisting_date >= listing_date)
);

CREATE TABLE IF NOT EXISTS observation_raw (
    entity_id      VARCHAR NOT NULL,
    event_date     DATE    NOT NULL,
    knowledge_date DATE    NOT NULL,
    value          DOUBLE,
    source         VARCHAR DEFAULT 'initial',
    revision_note  VARCHAR,
    PRIMARY KEY (entity_id, event_date, knowledge_date),
    CHECK (knowledge_date >= event_date)
);

CREATE TABLE IF NOT EXISTS labels_raw (
    entity_id  VARCHAR NOT NULL,
    event_date DATE    NOT NULL,
    value      DOUBLE,
    PRIMARY KEY (entity_id, event_date)
);

CREATE OR REPLACE VIEW observation_restated AS
SELECT DISTINCT ON (entity_id, event_date)
    entity_id,
    event_date,
    knowledge_date,
    value
FROM observation_raw
ORDER BY entity_id, event_date, knowledge_date DESC;

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

CREATE OR REPLACE MACRO universe_asof(cutoff) AS TABLE
    SELECT entity_id, sector
    FROM entities
    WHERE (listing_date   IS NULL OR listing_date   <= cutoff)
      AND (delisting_date IS NULL OR delisting_date  > cutoff);

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
