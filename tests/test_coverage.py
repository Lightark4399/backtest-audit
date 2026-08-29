"""Contracts for making both executed and omitted audits visible."""

from __future__ import annotations

from enum import Enum

import pytest

from audit.coverage import (
    AUDIT_REGISTRY,
    AuditRunState,
    AuditVerdict,
    SkipReasonCode,
)
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
