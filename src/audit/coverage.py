"""Closed, machine-readable contract for audit execution coverage.

Execution and conclusion are deliberately separate dimensions. ``SKIPPED``
means the check never ran; ``INCONCLUSIVE`` means it ran but the available data
could not support a verdict. Their remedies differ, so combining them would
erase information the caller needs to decide what to supply next.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AuditRunState(str, Enum):
    """Whether the audit's computation was attempted."""

    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"


class AuditVerdict(str, Enum):
    """Conclusion of a completed audit, independent of execution state."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SkipReasonCode(str, Enum):
    """Closed vocabulary for why an audit did not run."""

    MISSING_REQUIRED_INPUT = "missing_required_input"
    DISABLED_BY_CONFIG = "disabled_by_config"
    REQUIRES_EXTERNAL_EVIDENCE = "requires_external_evidence"


@dataclass(frozen=True)
class AuditSpec:
    """One shipped audit channel and the evidence it requires."""

    key: str
    display_name: str
    required_inputs: tuple[str, ...]


AUDIT_REGISTRY: tuple[AuditSpec, ...] = (
    AuditSpec("baseline", "baseline decomposition", ("prediction", "label")),
    AuditSpec("alignment", "alignment", ("prediction", "label", "event_date")),
    AuditSpec(
        "point_in_time",
        "point-in-time",
        ("event_time", "knowledge_time", "as_of_evidence"),
    ),
    AuditSpec("survivorship", "survivorship", ("dated_universe_membership",)),
    AuditSpec("grouping", "grouping", ("group_column",)),
    AuditSpec("significance", "serial-dependence significance", ("dated_ic_series",)),
    AuditSpec("protocol", "validation protocol", ("feature_columns:f_*",)),
    AuditSpec("execution", "execution timing", ("forward_return",)),
    AuditSpec("selection", "selection bias", ("all_candidate_return_series", "n_trials")),
)


@dataclass(frozen=True)
class SkipReason:
    """Why an audit was not executed."""

    code: SkipReasonCode
    reason: str


@dataclass(frozen=True)
class AuditCoverageEntry:
    """Execution state and, when completed, the resulting verdict."""

    spec: AuditSpec
    run_state: AuditRunState
    verdict: AuditVerdict | None
    skip_reason: SkipReason | None = None

    def to_dict(self) -> dict:
        return {
            "run_state": self.run_state.value,
            "verdict": self.verdict.value if self.verdict is not None else None,
            "reason_code": (
                self.skip_reason.code.value if self.skip_reason is not None else None
            ),
            "reason": self.skip_reason.reason if self.skip_reason is not None else None,
            "required_inputs": list(self.spec.required_inputs),
        }


def verdict_from_passed(passed: bool | None) -> AuditVerdict:
    """Translate the audit modules' existing tri-state verdict contract."""
    if passed is True:
        return AuditVerdict.PASS
    if passed is False:
        return AuditVerdict.FAIL
    return AuditVerdict.INCONCLUSIVE
