"""Human-readable audit report.

Design goal: a reader who knows nothing about this tool should be able to look at
the output for thirty seconds and come away with the right conclusion about
whether the result is trustworthy. That drives three choices:

* The decomposition is shown as a tree, so the free component sits visually
  underneath the headline number it explains rather than in a separate table the
  reader has to join mentally.
* Undefined values print as ``undefined`` with a reason, never as ``0.000``. The
  difference between "measured, found to be nil" and "not measurable" is exactly
  the kind of distinction that gets lost in summary statistics.
* Every report states its own scope (dates, entities, train boundary, label
  name). A number without its scope invites being quoted in a context where it
  is no longer true -- which is how a single-day figure ends up being cited as a
  general result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics.ic import ICSeries
from ..metrics.performance import annualisation_label
from ..metrics.significance import SignificanceResult

WIDTH = 78


def _fmt(value: float | None, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "undefined"
    return f"{value:+.{places}f}"


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _header(title: str) -> str:
    return f"\n{_rule('=')}\n{title}\n{_rule('=')}"


def format_scope(describe: dict) -> str:
    """Scope block: what data this report is about."""
    lines = [
        f"  evaluation scope       {describe.get('evaluation_scope', 'all')}",
        f"  label                 {describe['label_name']}",
        f"  entities              {describe['n_entities']}",
        f"  dates                 {describe['n_dates']}  "
        f"({describe['first_date']} .. {describe['last_date']})",
        f"  rows                  {describe['n_rows']:,}",
        f"  train boundary        {describe['train_end'] or 'NOT SET'}",
    ]
    if describe.get("rows_dropped_incomplete"):
        lines.append(
            f"  rows dropped          {describe['rows_dropped_incomplete']:,} "
            "(missing or non-finite prediction or label)"
        )
    return "\n".join(lines)


def format_audit_coverage(coverage: dict) -> str:
    """Declare which shipped audit channels did and did not run."""
    out = [_header("AUDIT COVERAGE"), ""]
    out.append(f"  Completed: {coverage['completed']} / {coverage['registered']}")
    out.append(f"  Skipped:   {coverage['skipped']} / {coverage['registered']}")
    skipped = [
        (key, entry)
        for key, entry in coverage["audits"].items()
        if entry["run_state"] == "SKIPPED"
    ]
    if skipped:
        out.append("")
        for key, entry in skipped:
            label = key.replace("_", "-")
            out.append(f"  {label:<24} SKIPPED")
            out.extend(_wrap(entry["reason"], indent=6))
    return "\n".join(out)


def format_ic_line(label: str, series: ICSeries, indent: int = 2, prefix: str = "") -> str:
    """One metric line: mean, dispersion, hit rate, and how many dates were usable."""
    pad = " " * indent
    name = f"{pad}{prefix}{label}"
    body = f"{name:<44}{_fmt(series.mean):>12}"

    if series.n_dates_used == 0:
        reason = "no usable cross-sections"
        if series.n_dates_undefined:
            reason = "correlation undefined (zero cross-sectional variance)"
        return f"{body}   [{reason}]"

    extras = [f"sd {series.std:.3f}" if np.isfinite(series.std) else "sd n/a"]
    if np.isfinite(series.hit_rate):
        extras.append(f"hit {series.hit_rate:.0%}")
    extras.append(f"n={series.n_dates_used}")
    if series.n_dates_undefined:
        extras.append(f"{series.n_dates_undefined} undefined")
    return f"{body}   [{', '.join(extras)}]"


def format_significance(sig: SignificanceResult, indent: int = 4) -> str:
    """Naive vs HAC inference, with the inflation factor made explicit."""
    pad = " " * indent
    if not np.isfinite(sig.hac_tstat):
        note = sig.notes[0] if sig.notes else "unavailable"
        return f"{pad}significance: {note}"

    lines = [
        f"{pad}t-stat (naive)        {sig.naive_tstat:>8.2f}",
        f"{pad}t-stat (Newey-West)   {sig.hac_tstat:>8.2f}   "
        f"[maxlags={sig.maxlags}, p={sig.hac_pvalue:.4f}]",
    ]
    if np.isfinite(sig.se_inflation):
        lines.append(
            f"{pad}SE inflation          {sig.se_inflation:>8.2f}x  "
            f"[lag-1 autocorr {sig.lag1_autocorr:+.2f}]"
        )
    if np.isfinite(sig.effective_n) and sig.information_loss > 0.01:
        lines.append(
            f"{pad}effective sample      {sig.effective_n:>8.0f} of {sig.n_obs} days"
            f"  [{sig.information_loss:.0%} lost to serial dependence]"
        )
    for n in sig.notes:
        # Notes can be long -- the effective-sample note runs past 150 characters
        # -- and an unwrapped line breaks the report's alignment in a narrow
        # terminal. Wrapped at the same width as every other block.
        wrapped = _wrap(f"note: {n}", indent=indent)
        lines.extend(wrapped)
    return "\n".join(lines)


def format_baseline_decomposition(
    raw: ICSeries,
    rank: ICSeries,
    baseline_table: pd.DataFrame,
    demeaned: ICSeries,
    incremental: ICSeries | None,
    demeaned_sig: SignificanceResult | None = None,
    incremental_sig: SignificanceResult | None = None,
) -> str:
    """The core exhibit: headline IC, what a naive predictor gets for free, and the remainder."""
    out = [_header("BASELINE DECOMPOSITION"), ""]
    out.append(format_ic_line("Raw IC (Pearson)", raw))
    out.append(format_ic_line("Raw IC (Spearman)", rank))
    out.append("")
    out.append("  Free score available without any model:")

    ordered = baseline_table.sort_values("mean", ascending=False, na_position="last")
    items = list(ordered.index)
    for i, name in enumerate(items):
        row = ordered.loc[name]
        connector = "└─ " if i == len(items) - 1 else "├─ "
        value = row["mean"]
        text = f"    {connector}{name:<38}"
        if pd.isna(value) or row.get("n_dates_used", 0) == 0:
            err = row.get("error")
            reason = err if isinstance(err, str) else "undefined (constant cross-section)"
            out.append(f"{text}{'undefined':>10}   [{reason}]")
        else:
            hit = row.get("hit_rate")
            extra = f"hit {hit:.0%}" if pd.notna(hit) else ""
            out.append(f"{text}{value:>+10.4f}   [{extra}]")

    out.append("")
    out.append("  After removing the per-entity level (training-period mean):")
    out.append(format_ic_line("Demeaned IC", demeaned, indent=4))
    if demeaned_sig is not None:
        out.append(format_significance(demeaned_sig, indent=6))

    if incremental is not None:
        out.append("")
        control = incremental.meta.get("control", "baseline")
        out.append(f"  Increment over strongest baseline ({control}), partial correlation:")
        out.append(format_ic_line("Incremental IC", incremental, indent=4))
        if incremental_sig is not None:
            out.append(format_significance(incremental_sig, indent=6))
        if incremental.meta.get("warning"):
            out.append(f"      WARNING: {incremental.meta['warning']}")
        if incremental.meta.get("note"):
            out.append(f"      note: {incremental.meta['note']}")

    return "\n".join(out)


def format_interpretation(
    raw: ICSeries, baseline_table: pd.DataFrame, demeaned: ICSeries
) -> str:
    """A plain-language verdict.

    Included because a table of numbers still permits the reader to draw the
    comfortable conclusion. Stating the implication explicitly removes that
    latitude. The thresholds are deliberately crude -- they are a prompt to look
    closer, not a certification.
    """
    out = [_header("READING"), ""]

    valid = baseline_table["mean"].dropna()
    best = float(valid.max()) if not valid.empty else float("nan")
    best_name = str(valid.idxmax()) if not valid.empty else "n/a"

    if np.isfinite(raw.mean) and np.isfinite(best):
        out.append(f"  Headline raw IC is {raw.mean:+.4f}.")
        out.append(
            f"  A naive predictor ({best_name}) reaches {best:+.4f} with no model at all."
        )
        # A baseline exceeding the model is a stronger finding than a baseline
        # merely accounting for most of it, so it gets said outright rather than
        # being reported as ">100% of the headline", which reads as a slip.
        if best >= raw.mean:
            out.append(
                "  The naive predictor BEATS the model outright: on this metric the model"
            )
            out.append("  adds nothing over doing no modelling at all.")
        elif raw.mean > 0 and best > 0:
            out.append(
                f"  -- that is {best / raw.mean:.0%} of the headline number, available for free."
            )
        elif raw.mean > 0:
            out.append(
                "  No naive baseline achieved a positive IC; the headline is not explained"
            )
            out.append("  by the free-score controls tested here.")
        else:
            out.append(
                "  The strongest naive baseline does not improve on the negative headline IC."
            )

    out.append("")
    if demeaned.n_dates_used == 0 or not np.isfinite(demeaned.mean):
        out.append("  Demeaned IC could not be computed; the decomposition is incomplete.")
    elif demeaned.mean <= -0.02:
        out.append(
            f"  Demeaned IC is {demeaned.mean:+.4f}: a negative association remains after"
        )
        out.append(
            "  removing the stable entity level. Check prediction and label sign conventions."
        )
        out.append(
            "  This is not evidence for an inverted trading signal: using a reversal requires"
        )
        out.append("  separate out-of-sample validation.")
    elif abs(demeaned.mean) < 0.02:
        out.append(
            "  Demeaned IC is indistinguishable from zero: once the stable per-entity"
        )
        out.append(
            "  level is removed, the prediction carries no information about deviations."
        )
        out.append("  The headline IC is measuring the level, not forecast skill.")
    elif demeaned.mean < 0.10:
        out.append(
            f"  Demeaned IC is {demeaned.mean:+.4f}: a small positive association beyond the level."
        )
        out.append("  Judge it against the free score above, not against the headline IC.")
    else:
        out.append(
            f"  Demeaned IC is {demeaned.mean:+.4f}: a substantial positive association remains"
        )
        out.append("  after removing each entity's typical level.")

    return "\n".join(out)


def format_alignment_audit(checks: list) -> str:
    """Alignment section: each perturbation, its effect, and the verdict.

    Verdicts are spelled out rather than reduced to PASS/FAIL flags, because the
    right reading of a shift test depends on how persistent the label is, and a
    bare flag would invite the reader to skip exactly the context that makes it
    interpretable.
    """
    out = [_header("ALIGNMENT AUDIT"), ""]
    out.append("  Does the result depend on the prediction being paired with the")
    out.append("  correct date's outcome?")
    out.append("")

    for c in checks:
        mark = {True: "PASS", False: "FAIL", None: "----"}[c.passed]
        out.append(f"  [{mark}] {c.name:<10}{c.description}")
        # The ratio is only shown when the check reached a verdict. On an
        # inconclusive check the baseline is near zero, so a percentage computed
        # from it is arithmetically valid but meaningless, and printing it would
        # invite the reader to draw a conclusion the check explicitly declined.
        show_ratio = c.passed is not None and np.isfinite(c.drop_ratio)
        out.append(
            f"         {c.baseline_ic:+.4f} -> {c.perturbed_ic:+.4f}"
            + (f"   ({c.drop_ratio:+.0%})" if show_ratio else "")
        )
        # Wrap the verdict so long explanations stay readable in a terminal.
        words, line = c.verdict.split(), "        "
        for w in words:
            if len(line) + len(w) + 1 > WIDTH - 2:
                out.append(line)
                line = "        " + w
            else:
                line = f"{line} {w}" if line.strip() else line + w
        if line.strip():
            out.append(line)
        out.append("")

    return "\n".join(out).rstrip()


def _wrap(text: str, indent: int = 8) -> list[str]:
    """Wrap a verdict so long explanations stay readable in a terminal."""
    pad = " " * indent
    out, line = [], pad
    for w in text.split():
        if len(line) + len(w) + 1 > WIDTH - 2:
            out.append(line)
            line = pad + w
        else:
            line = f"{line} {w}" if line.strip() else line + w
    if line.strip():
        out.append(line)
    return out


def format_group_decomposition(result) -> str:
    """Within-group vs between-group split of the pooled score."""
    mark = {True: "PASS", False: "FAIL", None: "----"}[result.passed]
    out = [_header("GROUP DECOMPOSITION"), ""]
    out.append(f"  Grouping key: {result.group_column}  ({result.n_groups} groups scored)")
    out.append("")
    out.append(f"  {'Pooled IC (whole cross-section)':<44}{result.pooled_ic:>+12.4f}")
    out.append(
        f"  {'Within-group IC (size-weighted)':<44}{result.within_ic_weighted:>+12.4f}"
    )
    out.append(
        f"  {'Within-group IC (unweighted)':<44}{result.within_ic_unweighted:>+12.4f}"
    )
    out.append(f"  {'Between-group IC (ranking groups)':<44}{result.between_ic:>+12.4f}")
    out.append(f"  {'Level effect (pooled - within)':<44}{result.level_effect:>+12.4f}")
    out.append("")
    out.append(f"  [{mark}]")
    out.extend(_wrap(result.verdict))

    table = result.per_group
    if len(table) <= 12:
        out.append("")
        out.append(f"    {'group':<16}{'n rows':>10}{'typical n':>12}{'IC':>10}")
        for g, row in table.iterrows():
            ic = "undefined" if pd.isna(row["ic"]) else f"{row['ic']:+.4f}"
            out.append(
                f"    {str(g):<16}{int(row['n_rows']):>10,}"
                f"{row['typical_cross_section']:>12.0f}{ic:>10}"
            )
    return "\n".join(out)


def format_survivorship(result) -> str:
    """Survivors-only vs point-in-time universe."""
    mark = {True: "PASS", False: "FAIL", None: "----"}[result.passed]
    out = [_header("SURVIVORSHIP"), ""]
    out.append(
        f"  {result.n_entities_total} entities, {result.n_entities_delisted} absent "
        f"at the end ({1 - result.survivor_rate:.1%} attrition)"
    )
    out.append("")
    out.append(
        f"  {'Point-in-time universe (demeaned IC)':<44}{result.pit_demeaned_ic:>+12.4f}"
    )
    out.append(
        f"  {'Survivors only (demeaned IC)':<44}{result.survivors_demeaned_ic:>+12.4f}"
    )
    out.append(f"  {'Gap attributable to survivorship':<44}{result.gap:>+12.4f}")
    out.append("")
    out.append(f"  [{mark}]")
    out.extend(_wrap(result.verdict))
    return "\n".join(out)


def format_pit(result) -> str:
    """Restated vs point-in-time data vintage."""
    mark = {True: "PASS", False: "FAIL", None: "----"}[result.passed]
    out = [_header("POINT-IN-TIME (DATA VINTAGE)"), ""]
    out.append("  Could these features have been computed at the time?")
    out.append("")
    out.append(
        f"  {'Restated data (corrections included)':<44}"
        f"{result.restated_demeaned_ic:>+12.4f}"
    )
    out.append(
        f"  {'As-of data (known at the time)':<44}{result.asof_demeaned_ic:>+12.4f}"
    )
    out.append(f"  {'Look-ahead advantage':<44}{result.gap:>+12.4f}")
    out.append("")
    out.append(
        f"  {result.n_revisions:,} of {result.n_observations:,} observations "
        f"corrected ({result.revision_rate:.1%}), mean lag "
        f"{result.mean_revision_lag_days:.0f} days"
    )
    out.append("")
    out.append(f"  [{mark}]")
    out.extend(_wrap(result.verdict))
    return "\n".join(out)


def format_protocol_comparison(comp) -> str:
    """Random vs ordered vs embargoed splitting, scored on the same model."""
    mark = {True: "PASS", False: "FAIL", None: "----"}[comp.passed]
    out = [_header("VALIDATION PROTOCOL"), ""]
    out.append("  Does the splitting scheme itself inflate the score?")
    out.append("")
    for r in comp.results:
        label = r.name.replace("_", " ")
        value = "undefined" if not np.isfinite(r.ic) else f"{r.ic:+.4f}"
        out.append(f"  {label:<30}{value:>12}   [{r.description}]")
    out.append("")
    out.append(f"  {'Inflation from random splitting':<30}{comp.inflation:>+12.4f}")
    if np.isfinite(comp.embargo_effect):
        out.append(f"  {'Removed by the embargo alone':<30}{comp.embargo_effect:>+12.4f}")
    out.append("")
    out.append(f"  [{mark}]")
    out.extend(_wrap(comp.verdict))
    return "\n".join(out)


def execution_lag_table(result) -> list[str]:
    """The per-lag IC and Sharpe rows, heading included.

    Shared with the demo. The demo previously rendered its own copy of these
    rows, which is how it went on printing "ann. Sharpe" over a per-period
    figure after the heading here had been corrected: two renderings of one
    number can always drift, and this one did.

    The Sharpe is per period. The execution audit receives no observation
    frequency, and its lag comparison is a ratio that sqrt-T scaling would
    leave unchanged in any case.
    """
    rows = [f"  {'execution delay':<22}{'IC':>12}{'Sharpe (per period)':>21}"]
    for r in result.results:
        ic = "undefined" if not np.isfinite(r.ic) else f"{r.ic:+.4f}"
        sh = "undefined" if not np.isfinite(r.sharpe) else f"{r.sharpe:+.4f}"
        rows.append(f"  lag {r.lag:<18}{ic:>12}{sh:>21}")
    return rows


def format_execution_timing(result) -> str:
    """Decay profile as execution is delayed."""
    mark = {True: "PASS", False: "FAIL", None: "----"}[result.passed]
    out = [_header("EXECUTION TIMING"), ""]
    out.append("  Could the signal have been traded when it was scored?")
    out.append("")
    out.extend(execution_lag_table(result))
    out.append("")
    if np.isfinite(result.decay_ratio):
        out.append(f"  {'retained at lag 1':<22}{result.decay_ratio:>11.0%}")
        out.append("")
    out.append(f"  [{mark}]")
    out.extend(_wrap(result.verdict))
    return "\n".join(out)


# Widest label (``approximate_selection_adjusted_sharpe_per_period``) is 48 and
# widest value (``winner_estimator``) is 16; one pair of widths for every row in
# the block keeps them aligned within WIDTH.
_SELECTION_LABEL_W = 49
_SELECTION_VALUE_W = 16


def selection_provenance_block(result) -> list[str]:
    """How the benchmark was computed, and what verdict that permits.

    Shared with the demo. Every row projects from
    ``SelectionProvenance.to_dict`` rather than being restated, and the note
    below them is written once here rather than once per surface -- the
    dataclass, this block, the JSON report and the coverage verdict are four
    renderings of one fact, and that shape has drifted before.
    """
    rows = [
        f"  {label:<{_SELECTION_LABEL_W}}{value:>{_SELECTION_VALUE_W}}"
        for label, value in result.provenance.as_rows()
    ]
    if result.provenance.method == "proxy":
        rows.extend(
            _wrap(
                "The Deflated Sharpe Ratio is defined against the variance of "
                "the Sharpe estimates across all trials. Without the Sharpe of "
                "every configuration examined, the winner's own estimator "
                "variance is substituted. The two agree only for independent, "
                "identically distributed trials; otherwise the direction of the "
                "error is not determined, so this cannot carry a PASS.",
                indent=2,
            )
        )
    return rows


def format_selection(result) -> str:
    """Observed Sharpe against what selection alone would produce.

    The observed figure is per period, and is labelled so. Reporting it under a
    bare "Sharpe" heading left it ambiguous against the annualised figure the
    surrounding narrative quoted, by a factor of sqrt(periods_per_year).

    The provenance block is projected from ``SelectionProvenance.to_dict`` and
    never restated here. The dataclass, this block, the JSON report and the
    coverage verdict are four renderings of one fact, which is the shape that
    has already drifted three times in this repository.
    """
    mark = {True: "PASS", False: "FAIL", None: "----"}[result.passed]
    out = [_header("SELECTION BIAS"), ""]
    out.append(
        f"  {'Observed Sharpe (per period)':<{_SELECTION_LABEL_W}}"
        f"{result.observed_sharpe:>+{_SELECTION_VALUE_W}.4f}"
    )
    periods = getattr(result, "periods_per_year", None)
    if periods is None:
        # Not the same claim as zero, and not a computation that failed: the
        # panel contract carries no observation frequency, so there is nothing
        # to annualise by.
        out.append(
            "  Sharpe (annualised)  "
            "NOT AVAILABLE -- observation frequency not supplied"
        )
    else:
        annualised = result.observed_sharpe_annualised
        label = f"Sharpe (annualised, {annualisation_label(periods)})"
        out.append(
            f"  {label:<{_SELECTION_LABEL_W}}"
            f"{annualised:>+{_SELECTION_VALUE_W}.4f}"
        )
    out.append(
        f"  {result.selection_adjusted_sharpe_key:<{_SELECTION_LABEL_W}}"
        f"{result.selection_adjusted_sharpe:>+{_SELECTION_VALUE_W}.4f}"
    )
    out.append(
        f"  {'Configurations examined':<{_SELECTION_LABEL_W}}"
        f"{result.n_trials:>{_SELECTION_VALUE_W}}"
    )
    out.append(
        f"  {'Expected maximum from selection alone':<{_SELECTION_LABEL_W}}"
        f"{result.expected_max_sharpe:>+{_SELECTION_VALUE_W}.4f}"
    )
    out.append(
        f"  {'Deflated probability':<{_SELECTION_LABEL_W}}"
        f"{result.deflated_probability:>{_SELECTION_VALUE_W}.3f}"
    )
    out.append("")
    out.extend(selection_provenance_block(result))
    out.append("")
    out.append(
        f"  return shape: skew {result.skew:+.2f}, kurtosis {result.kurtosis:.2f} "
        f"over {result.n_observations:.0f} observations"
    )
    out.append("")
    out.append(f"  [{mark}]")
    out.extend(_wrap(result.verdict))
    return "\n".join(out)


def render_report(
    scope: dict,
    raw: ICSeries,
    rank: ICSeries,
    baseline_table: pd.DataFrame,
    demeaned: ICSeries,
    incremental: ICSeries | None = None,
    demeaned_sig: SignificanceResult | None = None,
    incremental_sig: SignificanceResult | None = None,
    alignment_checks: list | None = None,
    group_result=None,
    survivorship_result=None,
    protocol_result=None,
    execution_result=None,
    selection_result=None,
    pit_result=None,
    title: str = "BACKTEST CREDIBILITY AUDIT",
    provenance: dict | None = None,
    audit_coverage: dict | None = None,
) -> str:
    """Assemble the full text report."""
    parts = [_rule("="), title.center(WIDTH), _rule("=")]
    parts.append("\nSCOPE")
    parts.append(format_scope(scope))
    if audit_coverage is not None:
        parts.append(format_audit_coverage(audit_coverage))
    parts.append(
        format_baseline_decomposition(
            raw, rank, baseline_table, demeaned, incremental, demeaned_sig, incremental_sig
        )
    )
    if alignment_checks:
        parts.append(format_alignment_audit(alignment_checks))
    if protocol_result is not None:
        parts.append(format_protocol_comparison(protocol_result))
    if execution_result is not None:
        parts.append(format_execution_timing(execution_result))
    if selection_result is not None:
        parts.append(format_selection(selection_result))
    if group_result is not None:
        parts.append(format_group_decomposition(group_result))
    if survivorship_result is not None:
        parts.append(format_survivorship(survivorship_result))
    if pit_result is not None:
        parts.append(format_pit(pit_result))
    parts.append(format_interpretation(raw, baseline_table, demeaned))

    if provenance:
        parts.append(_header("PROVENANCE"))
        parts.append("")
        key_width = max(22, max(len(str(k)) for k in provenance) + 2)
        for k, v in provenance.items():
            parts.append(f"  {k:<{key_width}}{v}")

    parts.append("")
    return "\n".join(parts)
