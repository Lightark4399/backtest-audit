"""Release metadata and public planning documents must describe the shipped tree."""

from __future__ import annotations

from pathlib import Path

import tomllib

from audit.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_release_version_is_v012():
    assert _project()["version"] == "0.1.2"


def test_distribution_and_console_script_names_are_unambiguous():
    """``backtest-audit`` on PyPI is an unrelated static-analysis tool.

    A reader who runs ``pip install backtest-audit`` gets someone else's project,
    so this distribution must keep a name of its own and the CLI must keep the
    name the README documents.
    """
    project = _project()
    assert project["name"] == "backtest-credibility-audit"
    assert project["scripts"] == {"btca": "audit.cli:main"}


def test_the_cli_announces_the_console_script_it_is_installed_as():
    """``prog`` is what usage text tells a user to type, so it cannot drift."""
    (script_name,) = _project()["scripts"]
    assert build_parser().prog == script_name


def test_report_provenance_reads_the_declared_distribution_name():
    """Provenance comes from distribution metadata, not the package name.

    A rename that misses this lookup does not fail: ``PackageNotFoundError`` is
    caught and every report claims ``auditor_version`` "unknown", which is false
    provenance rather than a crash.
    """
    source = (ROOT / "src" / "audit" / "run.py").read_text(encoding="utf-8")
    assert f'version("{_project()["name"]}")' in source


def test_readme_disambiguates_the_similarly_named_pypi_project():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "not published to PyPI" in readme
    assert "unrelated project" in readme
    assert "btca predictions.csv" in readme


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
