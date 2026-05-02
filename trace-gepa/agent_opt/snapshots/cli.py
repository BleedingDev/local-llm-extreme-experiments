"""CLI: python -m agent_opt.snapshots.cli <subcommand>."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

from .capture import capture
from .index import get, list_snapshots, meta_to_dict, purge_older_than


def _capture(a):
    print(json.dumps(meta_to_dict(capture(Path(a.workspace), prompt=a.prompt, label=a.label)), indent=2))
    return 0


def _list(a):
    for m in list_snapshots(limit=a.limit, label=a.label):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(m.captured_at))
        print(f"{m.id}  {ts}  {m.label or '-':10s}  {m.git_head[:8]}  {m.size_bytes:>9} B  {m.workspace}")
    return 0


def _show(a):
    m = get(a.id)
    if m is None:
        print(f"snapshot {a.id} not found", file=sys.stderr)
        return 1
    print(json.dumps(meta_to_dict(m), indent=2))
    return 0


def _purge(a):
    n = purge_older_than(days=a.older_than_days)
    print(f"purged {n} snapshot(s) older than {a.older_than_days} days")
    return 0


def _replay(a):
    print(f"replay TBD - see proposal trace-gepa/proposals/workflow_snapshots.md (id={a.id})")
    return 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bag-snapshot")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--workspace", default=".")
    c.add_argument("--prompt", default=None)
    c.add_argument("--label", default=None)
    c.set_defaults(func=_capture)
    ls = sub.add_parser("list")
    ls.add_argument("--limit", type=int, default=None)
    ls.add_argument("--label", default=None)
    ls.set_defaults(func=_list)
    sh = sub.add_parser("show")
    sh.add_argument("id")
    sh.set_defaults(func=_show)
    pu = sub.add_parser("purge")
    pu.add_argument("--older-than-days", type=float, default=30.0)
    pu.set_defaults(func=_purge)
    rp = sub.add_parser("replay")
    rp.add_argument("id")
    rp.set_defaults(func=_replay)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
