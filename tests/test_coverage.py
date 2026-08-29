"""Contracts for making both executed and omitted audits visible."""

from __future__ import annotations

import json
from enum import Enum

import pytest

from audit.audits.pit import PITResult
from audit.audits.selection import DeflatedSharpeResult
from audit.coverage import (
    AUDIT_REGISTRY,
    AuditRunState,
    AuditVerdict,
    SkipReasonCode,
)
from audit.demo import DEMO_AUDIT_CASES, write_demo_index
from audit.run import run_baseline_audit
from audit.synthetic import generate_panel

EXPECTED_AUDITS = {
    "baseline",
    "alignment",
    "point_in_time",
    "survivorship",
    "grouping",
    "significance",
    "protocol",
    "execution",
    "selection",
}


def _minimal_result():
    panel, _ = generate_panel(skill=0.4)
    return run_baseline_audit(panel)


def test_manifest_keys_equal_registered_audits():
    """A newly registered audit must not disappear from either report format."""
    registered = {spec.key for spec in AUDIT_REGISTRY}
    manifest = _minimal_result().to_dict()["audit_coverage"]

    assert registered == EXPECTED_AUDITS
    assert set(manifest["audits"]) == registered
    assert manifest["registered"] == len(registered)


def test_reason_codes_are_closed_enums_not_free_text():
    """Machine-readable skip reasons must not grow through ad-hoc strings."""
    assert issubclass(SkipReasonCode, Enum)
    assert {reason.value for reason in SkipReasonCode} == {
        "missing_required_input",
        "disabled_by_config",
        "requires_external_evidence",
    }


@pytest.mark.parametrize(
    ("audit", "expected_reason"),
    [
        ("protocol", SkipReasonCode.MISSING_REQUIRED_INPUT),
        ("execution", SkipReasonCode.MISSING_REQUIRED_INPUT),
        ("point_in_time", SkipReasonCode.REQUIRES_EXTERNAL_EVIDENCE),
        ("selection", SkipReasonCode.REQUIRES_EXTERNAL_EVIDENCE),
    ],
)
def test_missing_inputs_remain_visible_as_skipped(audit, expected_reason):
    entry = _minimal_result().to_dict()["audit_coverage"]["audits"][audit]

    assert entry["run_state"] == AuditRunState.SKIPPED.value
    assert entry["verdict"] is None
    assert entry["reason_code"] == expected_reason.value
    assert entry["reason"]
    assert entry["required_inputs"]


def test_skipped_and_inconclusive_are_not_interchangeable():
    """SKIPPED needs data; INCONCLUSIVE ran and may need more samples.

    The remedies are different. A skipped check can be made runnable by
    supplying its required evidence. An inconclusive check did run, but may
    need more observations or may simply be unanswerable on this dataset.
    Collapsing the states would leave the reader unable to tell what to do.
    """
    manifest = _minimal_result().to_dict()["audit_coverage"]["audits"]

    assert manifest["protocol"]["run_state"] == AuditRunState.SKIPPED.value
    assert manifest["protocol"]["verdict"] is None
    assert manifest["survivorship"]["run_state"] == AuditRunState.COMPLETED.value
    assert manifest["survivorship"]["verdict"] == AuditVerdict.INCONCLUSIVE.value


@pytest.mark.parametrize(
    ("kwargs", "audit"),
    [
        ({"run_alignment": False}, "alignment"),
        ({"run_survivorship": False}, "survivorship"),
        ({"run_protocol": False}, "protocol"),
        ({"group_column": None}, "grouping"),
    ],
)
def test_disabled_audit_is_distinct_from_missing_input(kwargs, audit):
    panel, _ = generate_panel(skill=0.4)
    result = run_baseline_audit(panel, **kwargs)
    entry = result.to_dict()["audit_coverage"]["audits"][audit]

    assert entry["run_state"] == AuditRunState.SKIPPED.value
    assert entry["reason_code"] == SkipReasonCode.DISABLED_BY_CONFIG.value


def test_text_report_declares_coverage_and_every_skip_reason():
    report = _minimal_result().to_text()

    assert "AUDIT COVERAGE" in report
    assert "Completed:" in report
    assert "Skipped:" in report
    for name in ("point-in-time", "protocol", "execution", "selection"):
        assert name in report.lower()
    assert "SKIPPED" in report


def _pit_result() -> PITResult:
    return PITResult(
        restated_ic=0.4,
        asof_ic=0.3,
        restated_demeaned_ic=0.2,
        asof_demeaned_ic=0.1,
        n_revisions=10,
        n_observations=100,
        revision_rate=0.1,
        mean_revision_size=0.02,
        mean_revision_lag_days=5.0,
        passed=False,
        verdict="FAIL: restated data inflated the result.",
    )


def _selection_result() -> DeflatedSharpeResult:
    return DeflatedSharpeResult(
        observed_sharpe=0.2,
        n_trials=42,
        n_observations=756,
        expected_max_sharpe=0.15,
        deflated_probability=0.4,
        skew=0.0,
        kurtosis=3.0,
        passed=False,
        verdict="FAIL: the selected result does not survive deflation.",
    )


@pytest.mark.parametrize(
    ("field", "audit", "value"),
    [
        ("pit", "point_in_time", _pit_result()),
        ("selection", "selection", _selection_result()),
    ],
)
def test_attached_external_result_updates_manifest_and_both_reports(field, audit, value):
    result = _minimal_result()
    setattr(result, field, value)

    serialised = result.to_dict()
    entry = serialised["audit_coverage"]["audits"][audit]
    assert entry["run_state"] == AuditRunState.COMPLETED.value
    assert entry["verdict"] == AuditVerdict.FAIL.value
    assert entry["reason_code"] is None
    assert serialised[audit] is not None
    assert audit.replace("_", "-") in result.to_text().lower()


def test_demo_index_covers_every_registered_audit(tmp_path):
    path = write_demo_index(tmp_path, DEMO_AUDIT_CASES)
    index = json.loads(path.read_text(encoding="utf-8"))

    assert set(index["audits"]) == EXPECTED_AUDITS
    assert index["registered"] == len(EXPECTED_AUDITS)
    assert all(entry["run_state"] == "COMPLETED" for entry in index["audits"].values())


def test_demo_index_rejects_an_invisible_registered_audit(tmp_path):
    incomplete = dict(DEMO_AUDIT_CASES)
    incomplete.pop("point_in_time")

    with pytest.raises(ValueError, match="point_in_time"):
        write_demo_index(tmp_path, incomplete)
