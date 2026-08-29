"""Release metadata and public planning documents must describe the shipped tree."""

from __future__ import annotations

from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_v012():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.1.2"


def test_plan_describes_the_current_test_and_module_status():
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    assert "Version 0.1.2" in plan
    assert "audit coverage manifest" in plan
    assert "Execution timing and selection bias" in plan
    assert "Feature freeze" in plan
    assert "A thin PnL layer" not in plan
    assert "179 tests" not in plan


def test_incident_log_records_the_coverage_contract_failure():
    notes = (ROOT / "AI_NOTES.md").read_text(encoding="utf-8")
    for incident in (15, 16, 17, 18):
        assert f"## Incident {incident}" in notes
    assert "business semantics" in notes
    assert "must not be weakened merely to make an implementation pass" in notes
    assert "undefined cannot masquerade as zero" in notes


def test_spec_requires_unrun_modules_to_be_declared():
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
    criterion = spec.split("**5.", 1)[1].split("**6.", 1)[0]
    assert "did not run" in criterion
    assert "reason" in criterion


def test_readme_uses_the_registry_count_without_stale_incident_or_test_totals():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "nine channels of inflation" in readme
    assert "the nine things that went wrong" not in readme
    assert "179 tests" not in readme
