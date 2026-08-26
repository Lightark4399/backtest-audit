"""Integration contracts for the audit orchestrator."""

from __future__ import annotations

import inspect

import pytest

from audit.run import run_baseline_audit
from audit.synthetic import generate_drifting_panel, generate_panel


def test_runner_exposes_no_in_sample_scope_switch():
    """A credibility verdict is out-of-sample; the runner must not advertise otherwise."""
    assert "scope" not in inspect.signature(run_baseline_audit).parameters


def test_runner_reports_and_computes_the_test_period_only():
    panel, _ = generate_panel(skill=0.4)
    result = run_baseline_audit(
        panel,
        run_alignment=False,
        run_survivorship=False,
        run_protocol=False,
    )

    assert result.config["evaluation_scope"] == "test"
    assert "scope" not in result.config
    assert result.scope["evaluation_scope"] == "test"
    assert result.scope["first_date"] > str(panel.train_end.date())
    assert all(date > panel.train_end for date in result.raw.values.index)
    assert all(date > panel.train_end for date in result.rank.values.index)
    assert all(date > panel.train_end for date in result.demeaned.values.index)


def test_provenance_never_guesses_the_audited_commit_from_cwd(tmp_path, monkeypatch):
    panel, _ = generate_panel(skill=0.4)
    monkeypatch.chdir(tmp_path)
    result = run_baseline_audit(
        panel,
        run_alignment=False,
        run_survivorship=False,
        run_protocol=False,
    )

    assert result.provenance["auditor_version"]
    assert result.provenance["build_commit"] == "unknown"
    assert result.provenance["audited_project_commit"] == "unknown"
    assert "git_commit" not in result.provenance
    assert "audited_project_commit  unknown" in result.to_text()


def test_audited_commit_is_recorded_only_when_the_caller_supplies_it():
    panel, _ = generate_panel(skill=0.4)
    result = run_baseline_audit(
        panel,
        audited_project_commit="strategy-repo@abc123",
        run_alignment=False,
        run_survivorship=False,
        run_protocol=False,
    )
    assert result.provenance["audited_project_commit"] == "strategy-repo@abc123"


def test_unexpected_protocol_errors_are_not_silently_omitted(monkeypatch):
    def broken_protocol(*args, **kwargs):
        raise RuntimeError("protocol implementation bug")

    monkeypatch.setattr("audit.run.compare_protocols", broken_protocol)
    with pytest.raises(RuntimeError, match="implementation bug"):
        run_baseline_audit(
            generate_drifting_panel(),
            run_alignment=False,
            run_survivorship=False,
        )
