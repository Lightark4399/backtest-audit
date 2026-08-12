-- Point-in-time access and feature construction.
--
-- Two views of the same history:
--
--   price_restated  the CURRENT best value for each (entity, event_date) --
--                   i.e. what a naive query against a mutable table returns
--   price_asof(d)   what was ACTUALLY KNOWN as of date d
--
-- Running the same pipeline against both and comparing the results is the
-- look-ahead test: if the restated version scores better, the pipeline depends
-- on information that did not exist when the forecast was supposedly made.

BEGIN;

-- ---------------------------------------------------------------------------
-- Restated view: latest version of every row, regardless of when it appeared
-- ---------------------------------------------------------------------------
--
-- DISTINCT ON is Postgres-specific and picks the first row per group given an
-- ORDER BY. It is used in preference to a window-function-plus-filter because it
-- can be satisfied directly by price_raw_pit_idx, and because it states the
-- intent ("one row per entity-date, the newest") in a single clause.

CREATE OR REPLACE VIEW price_restated AS
SELECT DISTINCT ON (entity_id, event_date)
    entity_id,
    event_date,
    knowledge_date,
    open, high, low, close, volume, source
FROM price_raw
ORDER BY entity_id, event_date, knowledge_date DESC;

COMMENT ON VIEW price_restated IS
    'Current best value per entity-date. Includes later corrections, so it must '
    'NOT be used to simulate a historical decision.';

-- ---------------------------------------------------------------------------
-- As-of view: the data as it stood on a given date
-- ---------------------------------------------------------------------------
--
-- A function rather than a view because the as-of date is a parameter. The
-- LATERAL subquery is the as-of join: for each entity-date, take the newest
-- version whose knowledge_date is at or before the cutoff. Rows first known
-- after the cutoff are excluded entirely -- they did not exist yet.

CREATE OR REPLACE FUNCTION price_asof(asof DATE)
RETURNS TABLE (
    entity_id      INTEGER,
    event_date     DATE,
    knowledge_date DATE,
    open           NUMERIC(18, 6),
    high           NUMERIC(18, 6),
    low            NUMERIC(18, 6),
    close          NUMERIC(18, 6),
    volume         NUMERIC(20, 4)
)
LANGUAGE sql STABLE AS $$
    SELECT DISTINCT ON (p.entity_id, p.event_date)
        p.entity_id, p.event_date, p.knowledge_date,
        p.open, p.high, p.low, p.close, p.volume
    FROM price_raw p
    WHERE p.knowledge_date <= asof
      -- A bar cannot inform a decision made before the bar's own date.
      AND p.event_date <= asof
    ORDER BY p.entity_id, p.event_date, p.knowledge_date DESC;
$$;

COMMENT ON FUNCTION price_asof(DATE) IS
    'Data as actually known on the given date. Use this to simulate historical '
    'decisions; compare against price_restated to detect look-ahead.';

-- ---------------------------------------------------------------------------
-- Survivorship-bias-free universe
-- ---------------------------------------------------------------------------
--
-- Membership is decided by listing/delisting dates, not by present existence.
-- An entity delisted mid-history remains in the universe for the dates it
-- actually traded, so its eventual failure stays in the sample.

CREATE OR REPLACE FUNCTION universe_asof(asof DATE)
RETURNS TABLE (entity_id INTEGER, ticker TEXT, sector TEXT)
LANGUAGE sql STABLE AS $$
    SELECT e.entity_id, e.ticker, e.sector
    FROM entities e
    WHERE (e.listing_date   IS NULL OR e.listing_date   <= asof)
      AND (e.delisting_date IS NULL OR e.delisting_date  > asof);
$$;

-- ---------------------------------------------------------------------------
-- Feature construction with strictly-past windows
-- ---------------------------------------------------------------------------
--
-- The critical detail is the frame clause:
--
--     ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
--
-- The default frame in a window function INCLUDES the current row. For a feature
-- meant to summarise history, including the current row leaks contemporaneous
-- information into what is supposed to be a lagged predictor -- and because the
-- result still looks like a plausible moving average, the error is easy to miss
-- in review. Ending the frame at "1 PRECEDING" makes the exclusion explicit.
--
-- tests/test_sql_windows.py asserts this property against a hand-computable
-- fixture, so a future edit that drops the frame clause fails the suite.

CREATE OR REPLACE VIEW daily_derived AS
WITH base AS (
    SELECT
        entity_id,
        event_date,
        open, high, low, close, volume,
        -- Parkinson-style intraday range: a scale-free volatility proxy with
        -- strong persistence, which is what makes it a good target for
        -- demonstrating the baseline trap.
        CASE WHEN high > 0 AND low > 0 AND high > low
             THEN ln(high / low)
             ELSE NULL
        END AS intraday_range
    FROM price_restated
)
SELECT
    entity_id,
    event_date,
    intraday_range,

    -- Lagged value: the persistence baseline's raw material.
    LAG(intraday_range, 1) OVER w AS intraday_range_lag1,

    -- Trailing mean over the previous 20 observations, CURRENT ROW EXCLUDED.
    AVG(intraday_range) OVER (
        PARTITION BY entity_id ORDER BY event_date
        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
    ) AS range_mean_20,

    STDDEV_SAMP(intraday_range) OVER (
        PARTITION BY entity_id ORDER BY event_date
        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
    ) AS range_sd_20,

    LAG(ln(NULLIF(volume, 0)), 1) OVER w AS log_volume_lag1,

    AVG(ln(NULLIF(volume, 0))) OVER (
        PARTITION BY entity_id ORDER BY event_date
        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
    ) AS log_volume_mean_20,

    -- Overnight and intraday returns, both strictly lagged.
    LAG(ln(close / NULLIF(open, 0)), 1) OVER w AS intraday_return_lag1
FROM base
WINDOW w AS (PARTITION BY entity_id ORDER BY event_date);

COMMENT ON VIEW daily_derived IS
    'Lagged features. All window frames end at 1 PRECEDING so no feature can '
    'contain same-day information.';

-- ---------------------------------------------------------------------------
-- Cross-sectional standardisation
-- ---------------------------------------------------------------------------
--
-- PERCENT_RANK within each event_date. Ranking is done per date rather than
-- pooled: a pooled rank would let the general level of a later period influence
-- an earlier one, which is a subtle full-sample leak.

CREATE OR REPLACE VIEW daily_cross_sectional_ranks AS
SELECT
    entity_id,
    event_date,
    intraday_range,
    PERCENT_RANK() OVER (PARTITION BY event_date ORDER BY range_mean_20)      AS pr_range_mean_20,
    PERCENT_RANK() OVER (PARTITION BY event_date ORDER BY log_volume_mean_20) AS pr_log_volume_mean_20,
    COUNT(*)      OVER (PARTITION BY event_date)                             AS cross_section_size
FROM daily_derived
WHERE range_mean_20 IS NOT NULL;

COMMIT;
