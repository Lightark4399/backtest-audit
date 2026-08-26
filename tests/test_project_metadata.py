"""Release metadata and public planning documents must describe the shipped tree."""

from __future__ import annotations

from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_v011():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.1.1"


def test_plan_describes_the_current_test_and_module_status():
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    assert "179 tests" in plan
    assert "Execution timing and selection bias" in plan
    assert "Feature freeze" in plan
    assert "A thin PnL layer" not in plan


def test_incident_log_records_the_three_release_shape_failures():
    notes = (ROOT / "AI_NOTES.md").read_text(encoding="utf-8")
    for incident in (15, 16, 17):
        assert f"## Incident {incident}" in notes
    assert "business semantics" in notes
    assert "must not be weakened merely to make an implementation pass" in notes
