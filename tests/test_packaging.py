"""Release-shape tests for resources needed at runtime."""

from __future__ import annotations

from importlib.resources import files

from audit.ingest.duckdb_store import BitemporalStore


def test_duckdb_schema_is_an_importable_package_resource():
    """A wheel cannot rely on the repository-level ``sql/`` directory."""
    schema = files("audit").joinpath("sql", "duckdb", "001_schema.sql")
    assert schema.is_file()
    assert "CREATE TABLE IF NOT EXISTS observation_raw" in schema.read_text(encoding="utf-8")


def test_bitemporal_store_initialises_from_packaged_schema(tmp_path, monkeypatch):
    """Runtime schema discovery must not depend on the caller's working directory."""
    monkeypatch.chdir(tmp_path)
    with BitemporalStore() as store:
        tables = store.con.execute("SHOW TABLES").fetchall()
    assert ("observation_raw",) in tables
