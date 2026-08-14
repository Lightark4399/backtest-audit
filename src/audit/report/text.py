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
            "(missing prediction or label)"
        )
    return "\n".join(lines)


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
    for n in sig.notes:
        lines.append(f"{pad}note: {n}")
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
        else:
            out.append(
                f"  -- that is {best / raw.mean:.0%} of the headline number, available for free."
            )

    out.append("")
    if demeaned.n_dates_used == 0:
        out.append("  Demeaned IC could not be computed; the decomposition is incomplete.")
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
            f"  Demeaned IC is {demeaned.mean:+.4f}: small but non-zero skill beyond the level."
        )
        out.append("  Judge it against the free score above, not against the headline IC.")
    else:
        out.append(
            f"  Demeaned IC is {demeaned.mean:+.4f}: the prediction contains real information"
        )
        out.append("  about deviations from each entity's typical level.")

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
    pit_result=None,
    title: str = "BACKTEST CREDIBILITY AUDIT",
    provenance: dict | None = None,
) -> str:
    """Assemble the full text report."""
    parts = [_rule("="), title.center(WIDTH), _rule("=")]
    parts.append("\nSCOPE")
    parts.append(format_scope(scope))
    parts.append(
        format_baseline_decomposition(
            raw, rank, baseline_table, demeaned, incremental, demeaned_sig, incremental_sig
        )
    )
    if alignment_checks:
        parts.append(format_alignment_audit(alignment_checks))
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
        for k, v in provenance.items():
            parts.append(f"  {k:<22}{v}")

    parts.append("")
    return "\n".join(parts)
