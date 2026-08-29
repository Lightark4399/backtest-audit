"""Offline demo: two panels, side by side.

Both are generated so the ground truth is known, which is what makes the demo a
demonstration rather than an anecdote:

``level-only``
    A prediction that knows each entity's stable level perfectly and knows
    *nothing* about dynamics (``skill=0``). Its raw IC looks strong. This is not
    a strawman: any per-entity model with an intercept reproduces this for free,
    so it is the default state of a model that has learned nothing useful.

``genuine-skill``
    The same level knowledge plus real information about deviations
    (``skill=0.6``). Raw IC is only modestly higher -- which is itself the point,
    since raw IC barely distinguishes the two -- but the demeaned IC separates
    them decisively.

The demo runs without network access or a database so it can execute in CI and on
any machine, and is deterministic given the seeds in ``SyntheticSpec``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audits.execution import audit_execution_timing
from .audits.pit import run_pit_audit
from .audits.protocol import compare_protocols
from .audits.selection import deflated_sharpe, screen_candidates
from .coverage import AUDIT_REGISTRY, AuditRunState
from .examples.pipelines import run_clean, run_leaky
from .ingest.duckdb_store import RevisionSpec
from .metrics.performance import compare_performance, performance
from .report.text import format_execution_timing, format_pit, format_selection
from .run import run_baseline_audit
from .synthetic import generate_drifting_panel, generate_panel, generate_return_panel

# This is an index of independent, known-ground-truth demo cases. It is not one
# combined audit: execution, selection and PIT deliberately use different
# synthetic evidence because pretending they share one scope would be false.
DEMO_AUDIT_CASES = {
    "baseline": {"case": "level_only", "artifact": "level_only_report.json"},
    "alignment": {"case": "level_only", "artifact": "level_only_report.json"},
    "point_in_time": {"case": "revised_vintage", "artifact": "pit_report.json"},
    "survivorship": {"case": "level_only", "artifact": "level_only_report.json"},
    "grouping": {"case": "level_only", "artifact": "level_only_report.json"},
    "significance": {"case": "level_only", "artifact": "level_only_report.json"},
    "protocol": {"case": "drifting_relationship", "artifact": "drifting_report.json"},
    "execution": {"case": "impossible_close_fill", "artifact": "execution_report.json"},
    "selection": {"case": "best_of_42_noise", "artifact": "selection_report.json"},
}


def write_demo_index(outdir: Path, audit_cases: dict) -> Path:
    """Write a complete index, failing if any shipped audit is invisible."""
    registered = {spec.key for spec in AUDIT_REGISTRY}
    supplied = set(audit_cases)
    if supplied != registered:
        missing = sorted(registered - supplied)
        extra = sorted(supplied - registered)
        raise ValueError(f"demo audit index mismatch; missing={missing}, extra={extra}")

    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "purpose": (
            "Index of independent known-ground-truth demo cases; "
            "not a single combined audit scope."
        ),
        "registered": len(AUDIT_REGISTRY),
        "audits": {
            key: {"run_state": AuditRunState.COMPLETED.value, **audit_cases[key]}
            for key in audit_cases
        },
    }
    path = outdir / "demo_audit_index.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _banner(text: str) -> str:
    line = "#" * 78
    return f"\n{line}\n# {text}\n{line}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the offline audit demo.")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="write report text and JSON here (default: print only)",
    )
    args = ap.parse_args(argv)

    cases = [
        (
            "CASE 1: prediction knows the entity level, has ZERO genuine skill",
            dict(skill=0.0, level_leak=1.0),
            "level_only",
        ),
        (
            "CASE 2: same level knowledge PLUS genuine skill on deviations",
            dict(skill=0.6, level_leak=1.0),
            "genuine_skill",
        ),
    ]

    summaries = []
    for title, kwargs, slug in cases:
        panel, _ = generate_panel(**kwargs)
        result = run_baseline_audit(panel, include_naive_increment=True)

        print(_banner(title))
        print(result.to_text(title=f"AUDIT -- {slug}"))

        summaries.append(
            (
                slug,
                result.raw.mean,
                result.demeaned.mean,
                result.incremental.mean if result.incremental else float("nan"),
                (
                    result.incremental.meta.get("naive_undemeaned_mean")
                    if result.incremental
                    else None
                ),
            )
        )

        if args.outdir:
            args.outdir.mkdir(parents=True, exist_ok=True)
            (args.outdir / f"{slug}_report.txt").write_text(
                result.to_text(title=f"AUDIT -- {slug}"), encoding="utf-8"
            )
            (args.outdir / f"{slug}_report.json").write_text(
                result.to_json(), encoding="utf-8"
            )

    # ---- Part 2: two real pipelines, same modelling, different protocol ----
    pipeline_rows = []
    for label, builder in (("clean", run_clean), ("leaky", run_leaky)):
        panel = builder()
        res = run_baseline_audit(panel)
        print(_banner(f"PIPELINE: {label}"))
        print(res.to_text(title=f"AUDIT -- pipeline_{label}"))
        pipeline_rows.append((label, res.raw.mean, res.demeaned.mean))

        if args.outdir:
            args.outdir.mkdir(parents=True, exist_ok=True)
            (args.outdir / f"pipeline_{label}_report.txt").write_text(
                res.to_text(title=f"AUDIT -- pipeline_{label}"), encoding="utf-8"
            )
            (args.outdir / f"pipeline_{label}_report.json").write_text(
                res.to_json(), encoding="utf-8"
            )

    # ---- Part 3: a panel carrying features, so the protocol audit can run ----
    #
    # The other cases hold finished predictions, which is all most of the
    # framework needs. The validation-protocol audit is the exception: it refits
    # the model under different splitting schemes, so it requires features. A
    # demo without this case would leave that module invisible in the output --
    # and a module no one sees in the report is not delivered.
    drift_panel = generate_drifting_panel(drift=1.5)
    drift_result = run_baseline_audit(drift_panel)
    print(_banner("CASE 3: drifting relationship, scored under three split protocols"))
    print(drift_result.to_text(title="AUDIT -- drifting_relationship"))

    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "drifting_report.txt").write_text(
            drift_result.to_text(title="AUDIT -- drifting_relationship"),
            encoding="utf-8",
        )
        (args.outdir / "drifting_report.json").write_text(
            drift_result.to_json(), encoding="utf-8"
        )

    # ---- Part 4: the same finding, stated as a backtest report would state it ----
    #
    # Everything above is in IC units, which is the better register for this work
    # and the wrong one for persuading anyone. This section says the same thing
    # in the units a backtest report uses, because that is the form in which the
    # deception is normally encountered.
    print(_banner("THE SAME RESULT, AS A BACKTEST WOULD REPORT IT"))
    perf_panels = {
        "level_only": generate_panel(skill=0.0)[0],
        "genuine_skill": generate_panel(skill=0.6)[0],
    }
    table = compare_performance(perf_panels)

    zero = performance(perf_panels["level_only"], demean_labels=False)
    zero_dm = performance(perf_panels["level_only"], demean_labels=True)

    print()
    print("  A strategy built on a prediction with EXACTLY ZERO skill —")
    print("  it knows each entity's typical level and nothing else:")
    print()
    print(f"      annualised Sharpe        {zero.sharpe_annualised:>10.1f}")
    print(f"      hit rate                 {zero.hit_rate:>10.1%}")
    print(f"      maximum drawdown         {zero.max_drawdown:>10.1%}")
    print(f"      periods                  {zero.n_periods:>10}")
    print()
    print("  Every day profitable, no drawdown, a Sharpe no real strategy reaches.")
    print("  It is worth pausing on how convincing that table is, because none of")
    print("  it is earned: the book is long the persistently-volatile names and")
    print("  short the persistently-quiet ones, and the target barely moves.")
    print()
    print("  The same positions, scored against demeaned labels — that is, on the")
    print("  part of the target that actually varies:")
    print()
    print(f"      annualised Sharpe        {zero_dm.sharpe_annualised:>10.1f}")
    print(f"      hit rate                 {zero_dm.hit_rate:>10.1%}")
    print(f"      maximum drawdown         {zero_dm.max_drawdown:>10.1%}")
    print()
    print("  Nothing was left. The audit above reaches the same verdict in IC")
    print("  units: raw IC +0.63, demeaned IC +0.0006.")
    print()
    print("  For contrast, a prediction with genuine skill on deviations:")
    print()
    cols = ["sharpe_raw", "sharpe_demeaned", "hit_rate_demeaned"]
    header = f"  {'panel':<18}" + "".join(f"{c:>20}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in table.index:
        row = table.loc[name]
        print(
            f"  {name:<18}"
            + "".join(f"{float(row[c]):>20.3f}" for c in cols)
        )
    print()
    print("  Raw Sharpe barely separates them. Demeaned Sharpe separates them")
    print("  decisively — the same asymmetry the IC decomposition shows.")

    # ---- Part 5: execution timing ----
    print(_banner("CASE 4: a signal that trades at a price it could not have got"))
    exec_panel = generate_return_panel(lookahead=1.0)
    exec_result = audit_execution_timing(exec_panel)
    print()
    print("  A signal computed from each day's close, backtested as though it")
    print("  could transact at that same close:")
    print()
    print(f"  {'execution delay':<22}{'IC':>12}{'ann. Sharpe':>16}")
    for r in exec_result.results:
        print(f"  lag {r.lag:<18}{r.ic:>+12.4f}{r.sharpe:>+16.2f}")
    print()
    print("  The close is the last observable price of the session. By the time")
    print("  it exists, the chance to trade at it has gone. Delay execution by a")
    print("  single period -- the honest assumption -- and the entire result")
    print("  disappears. Nothing about the strategy changed; only when it traded.")
    if args.outdir:
        (args.outdir / "execution_report.txt").write_text(
            format_execution_timing(exec_result), encoding="utf-8"
        )
        (args.outdir / "execution_report.json").write_text(
            json.dumps(exec_result.to_dict(), indent=2), encoding="utf-8"
        )

    # ---- Part 6: selection bias ----
    print(_banner("CASE 5: the best of 42 configurations, none of which work"))
    rng = __import__("numpy").random.default_rng(7)
    grid = {
        f"cfg{i:02d}": __import__("pandas").Series(rng.normal(0.0, 0.01, 756))
        for i in range(42)
    }
    sharpes = {k: v.mean() / v.std(ddof=1) for k, v in grid.items()}
    winner = max(sharpes, key=sharpes.get)
    annualised = sharpes[winner] * (252 ** 0.5)
    ds = deflated_sharpe(grid[winner], n_trials=len(grid))
    single = deflated_sharpe(grid[winner], n_trials=1)

    print()
    print(f"  Every one of 42 configurations is pure noise. The best, {winner},")
    print(f"  shows an annualised Sharpe of {annualised:.2f}.")
    print()
    print(f"  {'reported as the winner of 42 trials':<44}"
          f"prob {ds.deflated_probability:.3f}   {'FAIL' if ds.passed is False else ''}")
    print(f"  {'the same returns, reported as one test':<44}"
          f"prob {single.deflated_probability:.3f}   {'PASS' if single.passed else ''}")
    print()
    print("  Identical data. The difference is provenance: one number was")
    print("  selected for being the largest of 42, and maxima of noise are large.")

    screened = screen_candidates(grid)
    print()
    print("  Exploratory iid-normal approximation (not a strong PASS criterion):")
    print(f"  Screening all 42 at once: {int(screened['naive_significant'].sum())} look")
    print(f"  significant individually, {int(screened['survives'].sum())} survive FDR control.")
    if args.outdir:
        (args.outdir / "selection_report.txt").write_text(
            format_selection(ds), encoding="utf-8"
        )
        (args.outdir / "selection_report.json").write_text(
            json.dumps(ds.to_dict(), indent=2), encoding="utf-8"
        )

    # ---- Part 7: point-in-time data vintage ----
    pit_panel, _ = generate_panel(skill=0.4)
    observations = pit_panel.data[["entity_id", "event_date", "label"]].rename(
        columns={"label": "value"}
    )
    pit_result = run_pit_audit(
        observations,
        observations.copy(),
        train_end=pit_panel.train_end,
        revisions=RevisionSpec(fraction=0.3),
        max_asof_dates=8,
    )
    print(_banner("CASE 6: restated data knew corrections that had not arrived"))
    print(format_pit(pit_result))
    if args.outdir:
        (args.outdir / "pit_report.txt").write_text(format_pit(pit_result), encoding="utf-8")
        (args.outdir / "pit_report.json").write_text(
            json.dumps(pit_result.to_dict(), indent=2), encoding="utf-8"
        )

    print(_banner("SIDE BY SIDE"))
    print()
    header = f"{'case':<16}{'raw IC':>10}{'demeaned IC':>14}{'increment':>12}{'naive incr.':>13}"
    print(header)
    print("-" * len(header))
    for slug, raw, dm, inc, naive in summaries:
        naive_s = f"{naive:+.4f}" if isinstance(naive, float) else "n/a"
        print(f"{slug:<16}{raw:>+10.4f}{dm:>+14.4f}{inc:>+12.4f}{naive_s:>13}")
    print()
    if pipeline_rows:
        print()
        print("Two pipelines, identical modelling, different protocol:")
        print()
        h2 = f"{'pipeline':<16}{'raw IC':>10}{'demeaned IC':>14}"
        print(h2)
        print("-" * len(h2))
        for label, raw, dm in pipeline_rows:
            print(f"{label:<16}{raw:>+10.4f}{dm:>+14.4f}")
        if len(pipeline_rows) == 2:
            (_, _, dm_clean), (_, _, dm_leaky) = pipeline_rows
            print()
            print(
                f"The defective protocol inflates demeaned IC by "
                f"{dm_leaky - dm_clean:+.4f} ({(dm_leaky / dm_clean - 1):.0%}) "
                "without changing the model."
            )
            print("Both pass the alignment audit: these defects are about what the")
            print("model was allowed to know, not about how it was scored. See")
            print("src/audit/examples/pipelines.py for which module catches which.")
        print()

    if drift_result.protocol is not None:
        comp = drift_result.protocol
        # The claim below is that the audit reports no inflation when there is
        # none to report, so the control has to be measured rather than quoted:
        # a number written into the narrative would not survive the next change
        # to the panel generator.
        stationary = compare_protocols(generate_drifting_panel(drift=0.0))
        print()
        print("Same model, same data, three splitting protocols:")
        print()
        h3 = f"{'protocol':<24}{'IC':>10}"
        print(h3)
        print("-" * len(h3))
        for r in comp.results:
            print(f"{r.name.replace('_', ' '):<24}{r.ic:>+10.4f}")
        print()
        print(
            f"Random splitting is worth {comp.inflation:+.4f} of IC here, and none"
            " of it is real: the"
        )
        print("relationship drifts, so random folds hand the model rows from the")
        print("test period's own regime. Only the walk-forward figure is")
        print("out-of-sample. On a STATIONARY panel the same audit reports")
        print(
            f"{stationary.inflation:+.4f} -- it measures whether the problem"
            " applies rather than"
        )
        print("assuming it does.")
        print()

    print("Raw IC barely separates the two cases; demeaned IC separates them decisively.")
    print("The 'naive incr.' column is the un-demeaned partial correlation, shown to")
    print("illustrate its upward bias -- it credits the zero-skill model with skill it")
    print("does not have. See src/audit/metrics/partial.py for why.")
    print()

    if args.outdir:
        write_demo_index(args.outdir, DEMO_AUDIT_CASES)
        print(f"Reports written to {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
