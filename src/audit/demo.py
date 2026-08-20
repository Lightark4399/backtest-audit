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
from pathlib import Path

from .audits.protocol import compare_protocols
from .examples.pipelines import run_clean, run_leaky
from .run import run_baseline_audit
from .synthetic import generate_drifting_panel, generate_panel


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
                result.to_text(title=f"AUDIT -- {slug}")
            )
            (args.outdir / f"{slug}_report.json").write_text(result.to_json())

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
                res.to_text(title=f"AUDIT -- pipeline_{label}")
            )
            (args.outdir / f"pipeline_{label}_report.json").write_text(res.to_json())

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
            drift_result.to_text(title="AUDIT -- drifting_relationship")
        )
        (args.outdir / "drifting_report.json").write_text(drift_result.to_json())

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
        print(f"Reports written to {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
