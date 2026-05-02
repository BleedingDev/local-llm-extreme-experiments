"""CLI for per-tool calibration scorecards.

Examples:
  python -m agent_opt.calibration.cli single --results A.jsonl --tasks T.jsonl --out r.md
  python -m agent_opt.calibration.cli cross --results A.jsonl B.jsonl --tasks T.jsonl --out x.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agent_opt.calibration.scorecard import (
    compute_cross_model,
    compute_scorecard,
    render_cross_md,
    render_single_md,
)


def _cmd_single(args: argparse.Namespace) -> None:
    sc = compute_scorecard(Path(args.results), Path(args.tasks))
    title = args.title or f"Per-tool calibration scorecard: {Path(args.results).stem}"
    md = render_single_md(sc, title)
    Path(args.out).write_text(md)
    print(f"wrote {args.out} ({sc['n_total']} tasks, accuracy {sc['accuracy'] * 100:.1f}%)")


def _cmd_cross(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in args.results]
    cx = compute_cross_model(paths, Path(args.tasks))
    title = args.title or "Cross-model per-tool calibration scorecard"
    md = render_cross_md(cx, title)
    Path(args.out).write_text(md)
    print(f"wrote {args.out} ({len(paths)} models, {len(cx['pivot'])} tools)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agent_opt.calibration.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("single", help="single-model scorecard")
    s.add_argument("--results", required=True)
    s.add_argument("--tasks", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--title", default=None)
    s.set_defaults(func=_cmd_single)

    c = sub.add_parser("cross", help="cross-model scorecard")
    c.add_argument("--results", required=True, nargs="+")
    c.add_argument("--tasks", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--title", default=None)
    c.set_defaults(func=_cmd_cross)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
