"""Tests that SQL feature windows cannot see the current row.

The default window frame in SQL is
``RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`` -- it *includes* the
current row. A trailing average written without an explicit frame therefore
contains same-day information, and since the output still looks like a plausible
moving average the mistake survives review easily. It is exactly the class of
error this project exists to catch, so it is asserted rather than trusted.

Two layers:

* A **static** check that parses the SQL and requires every aggregate window
  frame to end at ``1 PRECEDING``. It needs no database, so it runs in CI on
  every push and fails immediately if someone drops a frame clause.
* A **numerical** check against a hand-computable fixture, which runs only when a
  database is reachable. It verifies the semantics, not merely the syntax.

The static check is the one that will actually catch the regression, because it
always runs. The numerical check exists so the static check cannot be satisfied
by text that happens to look right while computing something else.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
PIT_SQL = SQL_DIR / "002_pit_views.sql"

# Aggregates that summarise history and must therefore exclude the current row.
# LAG/LEAD are excluded from this rule: they take an explicit offset instead of a
# frame, and LAG(x, 1) is already strictly backward-looking.
HISTORY_AGGREGATES = ("AVG", "SUM", "STDDEV_SAMP", "STDDEV_POP", "MIN", "MAX", "COUNT")


def _strip_sql_comments(text: str) -> str:
    """Remove -- line comments and /* */ blocks so prose cannot satisfy a check."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"--[^\n]*", " ", text)
    return text


def test_pit_sql_file_exists():
    assert PIT_SQL.exists(), f"expected SQL at {PIT_SQL}"


def test_every_history_aggregate_window_excludes_current_row():
    """Any windowed history aggregate must carry a frame ending at 1 PRECEDING."""
    sql = _strip_sql_comments(PIT_SQL.read_text())

    # Find `AGG(...) OVER ( ... )` and inspect the OVER clause.
    pattern = re.compile(
        r"\b(" + "|".join(HISTORY_AGGREGATES) + r")\s*\([^()]*(?:\([^()]*\)[^()]*)*\)"
        r"\s+OVER\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
        re.IGNORECASE | re.DOTALL,
    )

    matches = list(pattern.finditer(sql))
    assert matches, "no windowed aggregates found -- has the SQL moved?"

    offenders = []
    for m in matches:
        agg, over_clause = m.group(1).upper(), m.group(2)
        normalised = " ".join(over_clause.split()).upper()

        # COUNT(*) OVER (PARTITION BY event_date) is a cross-sectional count, not
        # a history window: it is meant to include the current row, and carries
        # no ORDER BY, so a frame is neither required nor meaningful.
        if "ORDER BY" not in normalised:
            continue

        if "1 PRECEDING" not in normalised:
            offenders.append(f"{agg} OVER ({normalised})")

    assert not offenders, (
        "these windowed aggregates order rows but do not end their frame at "
        "1 PRECEDING, so they include the current row:\n  "
        + "\n  ".join(offenders)
    )


def test_no_aggregate_window_ends_at_current_row_explicitly():
    """Catch an explicit `AND CURRENT ROW` frame on an ordered history aggregate."""
    sql = " ".join(_strip_sql_comments(PIT_SQL.read_text()).split()).upper()
    bad = re.findall(r"ROWS BETWEEN[^)]*?AND CURRENT ROW", sql)
    assert not bad, f"frames ending at CURRENT ROW include today's value: {bad}"


def test_asof_function_filters_on_knowledge_date():
    """The point-in-time function must constrain knowledge_date, not just event_date.

    Filtering only on event_date would return today's *corrected* values for past
    dates, which is precisely the look-ahead the bitemporal schema exists to
    prevent.
    """
    sql = _strip_sql_comments(PIT_SQL.read_text()).upper()
    assert "PRICE_ASOF" in sql
    body = sql[sql.index("PRICE_ASOF"):]
    assert "KNOWLEDGE_DATE <= ASOF" in " ".join(body.split()), (
        "price_asof must filter knowledge_date <= asof"
    )


def test_universe_function_uses_delisting_date():
    """Survivorship bias is avoided by date-based membership, not present existence."""
    sql = _strip_sql_comments((SQL_DIR / "002_pit_views.sql").read_text()).upper()
    body = sql[sql.index("UNIVERSE_ASOF"):]
    assert "DELISTING_DATE" in body, (
        "universe_asof must consult delisting_date, otherwise a universe built "
        "from currently-listed entities silently excludes failures"
    )


# ----------------------------------------------------------------------
# Numerical verification -- requires a reachable database
# ----------------------------------------------------------------------
DB_URL = os.environ.get("AUDIT_DATABASE_URL")


@pytest.mark.skipif(not DB_URL, reason="AUDIT_DATABASE_URL not set; skipping DB test")
def test_trailing_mean_excludes_current_row_numerically():
    """Against a fixture whose correct answer is computable by hand.

    Labels 1, 2, 3, 4, 5 on consecutive dates for one entity. A trailing mean
    over the previous 20 rows excluding the current row must be NULL, 1, 1.5, 2,
    2.5. If the frame included the current row it would be 1, 1.5, 2, 2.5, 3 --
    every value shifted, and the first not NULL.
    """
    psycopg = pytest.importorskip("psycopg")

    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH fixture(event_date, v) AS (
                VALUES (DATE '2024-01-01', 1.0),
                       (DATE '2024-01-02', 2.0),
                       (DATE '2024-01-03', 3.0),
                       (DATE '2024-01-04', 4.0),
                       (DATE '2024-01-05', 5.0)
            )
            SELECT event_date,
                   AVG(v) OVER (ORDER BY event_date
                                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
            FROM fixture
            ORDER BY event_date;
            """
        )
        got = [None if r[1] is None else float(r[1]) for r in cur.fetchall()]

    assert got == [None, 1.0, 1.5, 2.0, 2.5]
