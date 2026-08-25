"""Audit orchestration.

Runs the metric layer over a panel and packages the results. Kept separate from
both the metrics (which stay pure functions over data) and the report (which
stays pure formatting) so that each can be tested without the others.

The JSON output carries provenance -- auditor version, optional build commit,
explicit audited-project identity, configuration and timestamp -- so a figure
can be traced without guessing from the caller's working directory.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import pandas as pd

from ._build_info import BUILD_COMMIT
from .audits.alignment import alignment_summary, run_alignment_audit
from .audits.execution import audit_execution_timing
from .audits.grouping import decompose_by_group
from .audits.protocol import compare_protocols
from .audits.survivorship import run_survivorship_audit
from .metrics.baselines import Baseline, default_baselines, evaluate_baselines, strongest_baseline
from .metrics.ic import ICSeries, demeaned_ic, rank_ic, raw_ic
from .metrics.partial import incremental_ic
from .metrics.significance import SignificanceResult, newey_west_tstat
from .panel import Panel
from .report.text import render_report


def _auditor_version() -> str:
    """Installed distribution version, or an honest marker for an unpackaged tree."""
    try:
        return version("backtest-audit")
    except PackageNotFoundError:
        return "unknown"


@dataclass
class AuditResult:
    """Everything one audit run produced."""

    scope: dict
    raw: ICSeries
    rank: ICSeries
    baseline_table: pd.DataFrame
    demeaned: ICSeries
    incremental: ICSeries | None = None
    demeaned_sig: SignificanceResult | None = None
    incremental_sig: SignificanceResult | None = None
    alignment: list | None = None
    grouping: object = None
    survivorship: object = None
    protocol: object = None
    execution: object = None
    # Not filled by run_baseline_audit: the Deflated Sharpe needs the return
    # series of all N candidates, and n_trials must be the number of configs
    # actually swept -- which cannot be inferred from a single panel. Inventing
    # a number is the exact behaviour this audit warns about. Callers compute
    # the result themselves and attach it here before rendering.
    selection: object = None
    provenance: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def to_text(self, title: str = "BACKTEST CREDIBILITY AUDIT") -> str:
        return render_report(
            scope=self.scope,
            raw=self.raw,
            rank=self.rank,
            baseline_table=self.baseline_table,
            demeaned=self.demeaned,
            incremental=self.incremental,
            demeaned_sig=self.demeaned_sig,
            incremental_sig=self.incremental_sig,
            alignment_checks=self.alignment,
            group_result=self.grouping,
            survivorship_result=self.survivorship,
            protocol_result=self.protocol,
            execution_result=self.execution,
            selection_result=self.selection,
            title=title,
            provenance=self.provenance,
        )

    def to_dict(self) -> dict:
        """Machine-readable form, suitable for CI assertions."""
        return {
            "provenance": self.provenance,
            "config": self.config,
            "scope": self.scope,
            "metrics": {
                "raw_ic": self.raw.to_dict(),
                "rank_ic": self.rank.to_dict(),
                "demeaned_ic": self.demeaned.to_dict(),
                "incremental_ic": self.incremental.to_dict() if self.incremental else None,
            },
            "baselines": json.loads(self.baseline_table.reset_index().to_json(orient="records")),
            "alignment": {
                "checks": [c.to_dict() for c in self.alignment] if self.alignment else None,
                "summary": alignment_summary(self.alignment) if self.alignment else None,
            },
            "grouping": self.grouping.to_dict() if self.grouping is not None else None,
            "protocol": self.protocol.to_dict() if self.protocol is not None else None,
            "execution": self.execution.to_dict() if self.execution is not None else None,
            "selection": self.selection.to_dict() if self.selection is not None else None,
            "survivorship": (
                self.survivorship.to_dict() if self.survivorship is not None else None
            ),
            "significance": {
                "demeaned_ic": self.demeaned_sig.to_dict() if self.demeaned_sig else None,
                "incremental_ic": (
                    self.incremental_sig.to_dict() if self.incremental_sig else None
                ),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def run_baseline_audit(
    panel: Panel,
    baselines: list[Baseline] | None = None,
    demean_method: str = "spearman",
    maxlags: int | None = None,
    include_naive_increment: bool = False,
    run_alignment: bool = True,
    group_column: str | None = "group",
    run_survivorship: bool = True,
    run_protocol: bool = True,
    return_column: str = "forward_return",
    audited_project_commit: str | None = None,
) -> AuditResult:
    """Run the baseline-decomposition audit on the out-of-sample test period.

    Parameters
    ----------
    include_naive_increment:
        Also compute the un-demeaned partial correlation. Off by default because
        it is a biased estimate of increment (see ``metrics/partial``); the demo
        turns it on deliberately to show the size of the bias.
    """
    baselines = baselines if baselines is not None else default_baselines()
    evaluation_scope = "test"

    raw = raw_ic(panel, scope=evaluation_scope)
    rnk = rank_ic(panel, scope=evaluation_scope)
    table = evaluate_baselines(panel, baselines, method="spearman", scope=evaluation_scope)
    dm = demeaned_ic(panel, method=demean_method)

    inc = None
    inc_sig = None
    best = strongest_baseline(table)
    if best is not None:
        control = next(b for b in baselines if b.name == best)
        inc = incremental_ic(panel, control, scope=evaluation_scope, demean=True)
        if inc.n_dates_used == 0:
            # The strongest baseline may be the level itself, which cannot be
            # controlled for twice. Fall back to the strongest baseline that
            # still has residual variation after demeaning, so the report shows a
            # usable increment rather than only an 'undefined'.
            for name in table.sort_values("mean", ascending=False).index:
                if name == best:
                    continue
                cand = next((b for b in baselines if b.name == name), None)
                if cand is None:
                    continue
                trial = incremental_ic(panel, cand, scope=evaluation_scope, demean=True)
                if trial.n_dates_used > 0:
                    inc = trial
                    break
        if inc.n_dates_used > 0:
            inc_sig = newey_west_tstat(inc.values, maxlags=maxlags)

    if include_naive_increment and best is not None:
        control = next(b for b in baselines if b.name == best)
        naive = incremental_ic(panel, control, scope=evaluation_scope, demean=False)
        if inc is not None:
            inc.meta["naive_undemeaned_mean"] = naive.mean

    dm_sig = newey_west_tstat(dm.values, maxlags=maxlags) if dm.n_dates_used else None

    # The alignment audit is part of the default run rather than an opt-in extra:
    # a decomposition of a number that was never correctly aligned would be a
    # precise analysis of an artefact.
    alignment = (
        run_alignment_audit(panel, scope=evaluation_scope) if run_alignment else None
    )

    # Group decomposition runs only when a grouping key is present. Absence is
    # not a failure -- many panels have no natural grouping -- so it is skipped
    # silently rather than reported as an unmet check.
    grouping = None
    if group_column and group_column in panel.data.columns:
        grouping = decompose_by_group(
            panel, group_col=group_column, scope=evaluation_scope
        )

    # Survivorship needs no extra input: attrition is visible in the panel
    # itself. On a balanced panel it correctly reports that the question cannot
    # be answered from the data.
    survivorship = (
        run_survivorship_audit(panel, scope=evaluation_scope)
        if run_survivorship
        else None
    )

    # The protocol audit refits the model under different splits, so it needs
    # feature columns. Panels carrying only finished predictions skip it rather
    # than failing: not every caller can supply features, and a missing input is
    # not an audit finding.
    protocol = None
    if run_protocol and any(c.startswith("f_") for c in panel.data.columns):
        # Expected insufficiency is represented by an INCONCLUSIVE
        # ProtocolComparison. Unexpected exceptions are implementation failures
        # and must surface rather than silently deleting an audit section.
        protocol = compare_protocols(panel)

    # Execution timing needs a return series. A panel carrying a non-tradeable
    # target skips it rather than producing a number about an execution that has
    # no meaning for that target.
    execution = None
    if return_column in panel.data.columns:
        try:
            execution = audit_execution_timing(
                panel, return_col=return_column, scope=evaluation_scope
            )
        except ValueError:
            # The audit raises ValueError when the return column is absent --
            # the expected case the guard above already screens for, so it is a
            # silent skip. Anything else is a real failure and should surface.
            execution = None

    provenance = {
        "auditor_version": _auditor_version(),
        "build_commit": BUILD_COMMIT or "unknown",
        "audited_project_commit": audited_project_commit or "unknown",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }
    config = {
        "evaluation_scope": evaluation_scope,
        "demean_method": demean_method,
        "maxlags": maxlags if maxlags is not None else "auto",
        "baselines": [b.name for b in baselines],
        "alignment_audit": run_alignment,
        "group_column": group_column,
        "survivorship_audit": run_survivorship,
        "protocol_audit": run_protocol,
        "return_column": return_column,
    }

    return AuditResult(
        scope=panel.describe(scope=evaluation_scope),
        raw=raw,
        rank=rnk,
        baseline_table=table,
        demeaned=dm,
        incremental=inc,
        demeaned_sig=dm_sig,
        incremental_sig=inc_sig,
        alignment=alignment,
        grouping=grouping,
        survivorship=survivorship,
        protocol=protocol,
        execution=execution,
        provenance=provenance,
        config=config,
    )
