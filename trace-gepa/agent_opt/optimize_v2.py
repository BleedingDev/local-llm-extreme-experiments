"""Thin wrapper around `agent_opt.optimize` that lets us point at v2 dataset/splits.

Agent V is concurrently editing optimize.py for a BAG-wiring fix. To avoid a
merge collision, this wrapper:
  1. Re-uses every helper / main flow from optimize.py.
  2. Adds two CLI flags `--dataset` and `--splits` defaulting to v1.
  3. Monkey-patches the module-level DATA / SPLITS constants before calling main.

Usage matches optimize.py exactly. Example:

    python -m agent_opt.optimize_v2 \
        --dataset trace-gepa/data/dataset_v2.jsonl \
        --splits  trace-gepa/data/splits_v2.json \
        --budget 600 --train-size 200 --val-size 80 \
        --seed-module default --run-name v2_big
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_PARENT = _HERE.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from agent_opt import optimize as _opt  # noqa: E402


def _split_argv() -> tuple[Path, Path, list[str]]:
    """Pull --dataset / --splits out of sys.argv and return them + the rest.

    We can't just call argparse twice because the inner parser in optimize.main
    will choke on unknown flags. So we strip them before delegating.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--dataset",
        default=str(_opt.ROOT / "data" / "dataset.jsonl"),
        help="Path to dataset JSONL (default: v1).",
    )
    pre.add_argument(
        "--splits",
        default=str(_opt.ROOT / "data" / "splits.json"),
        help="Path to splits JSON (default: v1).",
    )
    args, rest = pre.parse_known_args()
    return Path(args.dataset).resolve(), Path(args.splits).resolve(), rest


def main() -> int:
    dataset, splits, rest = _split_argv()
    if not dataset.exists():
        print(f"[optimize_v2] dataset not found: {dataset}", file=sys.stderr)
        return 2
    if not splits.exists():
        print(f"[optimize_v2] splits not found: {splits}", file=sys.stderr)
        return 2

    # Patch module constants before optimize.main() reads them.
    _opt.DATA = dataset
    _opt.SPLITS = splits

    # Replace argv so optimize.main()'s argparse only sees its own flags.
    sys.argv = [sys.argv[0], *rest]
    print(f"[optimize_v2] dataset={dataset}")
    print(f"[optimize_v2] splits={splits}")
    return _opt.main()


if __name__ == "__main__":
    raise SystemExit(main())
