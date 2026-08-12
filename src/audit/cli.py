"""Command line entry point.

Accepts a CSV of predictions and labels so the tool can audit results produced by
any pipeline, in any language, without integration work. The panel contract in
``panel.py`` is the only interface: four columns and a training boundary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .panel import Panel, PanelError
from .run import run_baseline_audit


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="backtest-audit",
        description=(
            "Audit a backtest result: decompose its IC into the part any naive "
            "baseline achieves for free and the part attributable to the model."
        ),
    )
    ap.add_argument(
        "csv",
        type=Path,
        help="CSV with columns entity_id, event_date, prediction, label",
    )
    ap.add_argument(
        "--train-end",
        required=True,
        help=(
            "last date of the training period (YYYY-MM-DD). Required: metrics "
            "that demean by a per-entity mean must compute it from training data "
            "only, and there is no safe default."
        ),
    )
    ap.add_argument("--label-name", default="label", help="description of the target")
    ap.add_argument(
        "--scope",
        default="test",
        choices=("test", "train", "all"),
        help="which period to evaluate (default: test, i.e. after train-end)",
    )
    ap.add_argument(
        "--method",
        default="spearman",
        choices=("spearman", "pearson"),
        help="correlation used for the demeaned IC (default: spearman)",
    )
    ap.add_argument(
        "--maxlags",
        type=int,
        default=None,
        help="Newey-West bandwidth (default: automatic, 4*(T/100)^(2/9))",
    )
    ap.add_argument("--json", type=Path, default=None, help="also write JSON here")
    ap.add_argument(
        "--show-naive-increment",
        action="store_true",
        help="also report the un-demeaned partial correlation (biased; for comparison)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.csv.exists():
        print(f"error: no such file: {args.csv}", file=sys.stderr)
        return 2

    try:
        frame = pd.read_csv(args.csv)
        panel = Panel.from_frame(
            frame, train_end=args.train_end, label_name=args.label_name
        )
    except PanelError as exc:
        # Contract violations are user-fixable input problems, so they print a
        # clean message rather than a traceback.
        print(f"error: invalid panel: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: could not read {args.csv}: {exc}", file=sys.stderr)
        return 2

    result = run_baseline_audit(
        panel,
        scope=args.scope,
        demean_method=args.method,
        maxlags=args.maxlags,
        include_naive_increment=args.show_naive_increment,
    )

    print(result.to_text())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(result.to_json())
        print(f"JSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
