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

from .examples.pipelines import run_clean, run_leaky
from .run import run_baseline_audit
from .synthetic import generate_panel


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
