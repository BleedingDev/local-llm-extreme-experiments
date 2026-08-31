#!/usr/bin/env python3
"""Run a small Hermes local-model coding evaluation suite."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts" / "hermes-coding-eval" / time.strftime("%Y%m%d-%H%M%S")
HERMES = Path.home() / ".local" / "bin" / "hermes"
PROFILE = os.environ.get("HERMES_EVAL_PROFILE", "hermeslocalqwen")
MODEL = "majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit"


@dataclass
class Task:
    name: str
    summary: str
    files: dict[str, str]
    test_cmd: list[str]
    hidden_cmd: list[str]
    prompt: str
    max_turns: int = 18
    timeout_s: int = 900


def dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


TASKS = [
    Task(
        name="py_slug_unicode",
        summary="Single-file Python bugfix: robust slugify behavior for punctuation, accents, and whitespace.",
        files={
            "slugify.py": dedent(
                r"""
                import re


                def slugify(title: str, max_length: int = 50) -> str:
                    text = title.lower().strip()
                    text = re.sub(r"\s+", "-", text)
                    text = re.sub(r"[^a-z0-9-]", "", text)
                    if len(text) > max_length:
                        text = text[:max_length]
                    return text
                """
            ),
            "test_slugify.py": dedent(
                r"""
                import unittest
                from slugify import slugify


                class SlugifyTests(unittest.TestCase):
                    def test_basic(self):
                        self.assertEqual(slugify("Hello, World!"), "hello-world")

                    def test_collapses_separators(self):
                        self.assertEqual(slugify("  A --- B___C  "), "a-b-c")

                    def test_unicode_accents(self):
                        self.assertEqual(slugify("Český Krumlov déjà vu"), "cesky-krumlov-deja-vu")

                    def test_truncates_without_trailing_hyphen(self):
                        self.assertEqual(slugify("alpha beta gamma", max_length=12), "alpha-beta")


                if __name__ == "__main__":
                    unittest.main()
                """
            ),
        },
        test_cmd=["python3", "-m", "unittest", "-v"],
        hidden_cmd=[
            "python3",
            "-c",
            "from slugify import slugify; assert slugify('Rock & Roll!!!') == 'rock-roll'; assert slugify('---') == ''; assert slugify('  déjà---vu  ', 9) == 'deja-vu'",
        ],
        prompt="Fix slugify.py so all tests pass. Preserve the public function signature. Run the tests before finishing.",
    ),
    Task(
        name="py_intervals",
        summary="Algorithmic implementation: merge half-open intervals and reject invalid input.",
        files={
            "intervals.py": dedent(
                r"""
                def merge_intervals(intervals):
                    # Return sorted, merged half-open intervals [start, end).
                    merged = []
                    for start, end in intervals:
                        if not merged or start > merged[-1][1]:
                            merged.append([start, end])
                        else:
                            merged[-1][1] = end
                    return [tuple(item) for item in merged]
                """
            ),
            "test_intervals.py": dedent(
                r"""
                import unittest
                from intervals import merge_intervals


                class IntervalTests(unittest.TestCase):
                    def test_sorts_before_merging(self):
                        self.assertEqual(merge_intervals([(5, 8), (1, 3), (2, 6)]), [(1, 8)])

                    def test_touching_half_open_intervals_merge(self):
                        self.assertEqual(merge_intervals([(1, 2), (2, 3), (8, 9)]), [(1, 3), (8, 9)])

                    def test_nested_interval_keeps_larger_end(self):
                        self.assertEqual(merge_intervals([(1, 10), (3, 4)]), [(1, 10)])

                    def test_invalid_interval_raises(self):
                        with self.assertRaises(ValueError):
                            merge_intervals([(3, 3)])


                if __name__ == "__main__":
                    unittest.main()
                """
            ),
        },
        test_cmd=["python3", "-m", "unittest", "-v"],
        hidden_cmd=[
            "python3",
            "-c",
            "from intervals import merge_intervals; assert merge_intervals([]) == []; assert merge_intervals([(0, 1), (-2, 0), (10, 12)]) == [(-2, 1), (10, 12)]",
        ],
        prompt="Implement merge_intervals in intervals.py correctly for unsorted half-open intervals. Validate that every interval has start < end and raise ValueError otherwise. Run the tests before finishing.",
    ),
    Task(
        name="py_parser",
        summary="Parser task: parse simple key-value config with comments, quotes, integers, booleans, and duplicate detection.",
        files={
            "mini_config.py": dedent(
                r"""
                def parse_config(text: str) -> dict:
                    result = {}
                    for line in text.splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        key, value = line.split("=")
                        result[key.strip()] = value.strip()
                    return result
                """
            ),
            "test_mini_config.py": dedent(
                r"""
                import unittest
                from mini_config import parse_config


                class MiniConfigTests(unittest.TestCase):
                    def test_types_and_comments(self):
                        cfg = parse_config('host = "localhost" # comment\nport = 8080\ndebug = true\n')
                        self.assertEqual(cfg, {"host": "localhost", "port": 8080, "debug": True})

                    def test_hash_inside_quotes(self):
                        self.assertEqual(parse_config("token = 'abc#123'\n"), {"token": "abc#123"})

                    def test_duplicate_key_raises(self):
                        with self.assertRaises(ValueError):
                            parse_config("a=1\na=2\n")

                    def test_bad_line_raises(self):
                        with self.assertRaises(ValueError):
                            parse_config("not a setting\n")


                if __name__ == "__main__":
                    unittest.main()
                """
            ),
        },
        test_cmd=["python3", "-m", "unittest", "-v"],
        hidden_cmd=[
            "python3",
            "-c",
            "from mini_config import parse_config; assert parse_config('x=false\\ny=42\\nz=plain\\n') == {'x': False, 'y': 42, 'z': 'plain'}",
        ],
        prompt="Fix mini_config.py. It should parse a small key=value format with # comments outside quotes, single or double quoted strings, integers, booleans true/false, duplicate-key errors, and invalid-line errors. Run tests.",
    ),
    Task(
        name="py_multifile_cli",
        summary="Multi-file Python CLI task: wire CSV loader, aggregation, and command-line JSON output.",
        files={
            "sales/__init__.py": "",
            "sales/io.py": dedent(
                r"""
                import csv


                def load_sales(path):
                    with open(path, newline="") as handle:
                        return list(csv.DictReader(handle))
                """
            ),
            "sales/summary.py": dedent(
                r"""
                def summarize(rows):
                    return {}
                """
            ),
            "sales/cli.py": dedent(
                r"""
                import argparse


                def main(argv=None):
                    parser = argparse.ArgumentParser()
                    parser.add_argument("csv_path")
                    args = parser.parse_args(argv)
                    print(args.csv_path)


                if __name__ == "__main__":
                    main()
                """
            ),
            "sample.csv": "region,amount\nEU,10.50\nUS,5\nEU,2.25\nAPAC,7\n",
            "test_sales.py": dedent(
                r"""
                import json
                import subprocess
                import sys
                import unittest
                from decimal import Decimal

                from sales.io import load_sales
                from sales.summary import summarize


                class SalesTests(unittest.TestCase):
                    def test_summarize_by_region(self):
                        rows = load_sales("sample.csv")
                        self.assertEqual(summarize(rows), {"APAC": Decimal("7.00"), "EU": Decimal("12.75"), "US": Decimal("5.00")})

                    def test_cli_outputs_sorted_json(self):
                        out = subprocess.check_output([sys.executable, "-m", "sales.cli", "sample.csv"], text=True)
                        self.assertEqual(json.loads(out), {"APAC": "7.00", "EU": "12.75", "US": "5.00"})


                if __name__ == "__main__":
                    unittest.main()
                """
            ),
        },
        test_cmd=["python3", "-m", "unittest", "-v"],
        hidden_cmd=[
            "python3",
            "-c",
            "from decimal import Decimal; from sales.summary import summarize; assert summarize([{'region':'A','amount':'1.005'}, {'region':'A','amount':'2'}]) == {'A': Decimal('3.01')}",
        ],
        prompt="Make the sales package work. summarize(rows) must total amount by region using Decimal money arithmetic rounded to cents, and python -m sales.cli sample.csv must print sorted JSON with string amounts. Run tests.",
        max_turns=22,
    ),
    Task(
        name="js_lru_cache",
        summary="JavaScript Node task: implement an LRU cache with updates, eviction, and validation.",
        files={
            "package.json": dedent(
                r"""
                {
                  "type": "module",
                  "scripts": {
                    "test": "node --test"
                  }
                }
                """
            ),
            "lru-cache.js": dedent(
                r"""
                export class LRUCache {
                  constructor(capacity) {
                    this.capacity = capacity;
                    this.items = new Map();
                  }

                  get(key) {
                    return this.items.get(key) ?? null;
                  }

                  set(key, value) {
                    this.items.set(key, value);
                  }
                }
                """
            ),
            "lru-cache.test.js": dedent(
                r"""
                import test from "node:test";
                import assert from "node:assert/strict";
                import { LRUCache } from "./lru-cache.js";

                test("evicts least recently used key", () => {
                  const cache = new LRUCache(2);
                  cache.set("a", 1);
                  cache.set("b", 2);
                  assert.equal(cache.get("a"), 1);
                  cache.set("c", 3);
                  assert.equal(cache.get("b"), null);
                  assert.equal(cache.get("c"), 3);
                });

                test("updating a key refreshes recency", () => {
                  const cache = new LRUCache(2);
                  cache.set("a", 1);
                  cache.set("b", 2);
                  cache.set("a", 10);
                  cache.set("c", 3);
                  assert.equal(cache.get("a"), 10);
                  assert.equal(cache.get("b"), null);
                });

                test("invalid capacity throws", () => {
                  assert.throws(() => new LRUCache(0), /capacity/i);
                });
                """
            ),
        },
        test_cmd=["npm", "test"],
        hidden_cmd=[
            "node",
            "--input-type=module",
            "-e",
            "import { LRUCache } from './lru-cache.js'; const c = new LRUCache(1); c.set('x', undefined); if (c.get('x') !== undefined) throw new Error('undefined values must be stored'); c.set('y', 2); if (c.get('x') !== null) throw new Error('eviction failed');",
        ],
        prompt="Implement LRUCache in lru-cache.js. get should return null for missing keys but must preserve stored undefined values. set should update recency and evict the least recently used item. Reject non-positive capacities. Run npm test.",
    ),
    Task(
        name="py_async_limiter",
        summary="Async/concurrency task: implement a rate limiter that spaces task starts and preserves result order.",
        files={
            "limiter.py": dedent(
                r"""
                import asyncio


                async def run_limited(callables, *, interval):
                    results = []
                    for fn in callables:
                        results.append(await fn())
                    return results
                """
            ),
            "test_limiter.py": dedent(
                r"""
                import asyncio
                import time
                import unittest
                from limiter import run_limited


                class LimiterTests(unittest.IsolatedAsyncioTestCase):
                    async def test_starts_are_spaced_and_results_ordered(self):
                        starts = []

                        def make_fn(value, delay):
                            async def fn():
                                starts.append(time.perf_counter())
                                await asyncio.sleep(delay)
                                return value
                            return fn

                        result = await run_limited(
                            [make_fn("a", 0.08), make_fn("b", 0.01), make_fn("c", 0.01)],
                            interval=0.05,
                        )

                        self.assertEqual(result, ["a", "b", "c"])
                        self.assertGreaterEqual(starts[1] - starts[0], 0.045)
                        self.assertGreaterEqual(starts[2] - starts[1], 0.045)
                        self.assertLess(starts[-1] - starts[0], 0.13)

                    async def test_invalid_interval(self):
                        with self.assertRaises(ValueError):
                            await run_limited([], interval=-1)


                if __name__ == "__main__":
                    unittest.main()
                """
            ),
        },
        test_cmd=["python3", "-m", "unittest", "-v"],
        hidden_cmd=[
            "python3",
            "-c",
            "import asyncio\nfrom limiter import run_limited\nasync def main():\n    async def a():\n        return 1\n    assert await run_limited([a], interval=0) == [1]\nasyncio.run(main())",
        ],
        prompt="Fix limiter.py. run_limited receives async zero-argument callables. It should start them no faster than one every interval seconds, allow overlap, preserve result order, and raise ValueError for negative interval. Run tests.",
        max_turns=20,
    ),
]


def write_files(task_dir: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = task_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path, timeout: int = 120) -> dict:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "duration_s": round(time.perf_counter() - start, 3),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "duration_s": round(time.perf_counter() - start, 3),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout": True,
        }


def run_hermes(task: Task, task_dir: Path) -> dict:
    prompt = dedent(
        f"""
        You are evaluating a local coding model through Hermes Agent.

        Work only inside this task directory:
        {task_dir}

        Task:
        {task.prompt}

        Requirements:
        - Inspect the files before editing.
        - Make the minimal code changes needed.
        - Use terminal tests to verify.
        - Stop when the implementation is correct.
        """
    )
    log_path = task_dir / "hermes.log"
    cmd = [
        str(HERMES),
        "-p",
        PROFILE,
        "--yolo",
        "chat",
        "-Q",
        "-q",
        prompt,
        "--max-turns",
        str(task.max_turns),
        "--toolsets",
        "file,terminal",
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=task_dir,
            text=True,
            capture_output=True,
            timeout=task.timeout_s,
            env={
                **os.environ,
                "HERMES_API_TIMEOUT": "900",
                "HERMES_STREAM_READ_TIMEOUT": "90",
                "HERMES_SESSION_SOURCE": "local-coding-eval",
            },
        )
        output = proc.stdout + proc.stderr
        log_path.write_text(output, encoding="utf-8")
        return {
            "returncode": proc.returncode,
            "duration_s": round(time.perf_counter() - start, 3),
            "timeout": False,
            "log_path": str(log_path.relative_to(ROOT)),
            "session_ids": sorted(set(part for part in output.split() if part.startswith("2026"))),
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(output, encoding="utf-8")
        return {
            "returncode": None,
            "duration_s": round(time.perf_counter() - start, 3),
            "timeout": True,
            "log_path": str(log_path.relative_to(ROOT)),
            "session_ids": sorted(set(part for part in output.split() if part.startswith("2026"))),
            "output_tail": output[-4000:],
        }


def main() -> int:
    if not HERMES.exists():
        raise SystemExit(f"Hermes binary not found: {HERMES}")
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True)

    results = []
    for task in TASKS:
        task_dir = RUN_ROOT / task.name
        task_dir.mkdir(parents=True)
        write_files(task_dir, task.files)

        before = run_cmd(task.test_cmd, task_dir, timeout=120)
        hermes = run_hermes(task, task_dir)
        after = run_cmd(task.test_cmd, task_dir, timeout=180)
        hidden = run_cmd(task.hidden_cmd, task_dir, timeout=120)
        passed = after["returncode"] == 0 and hidden["returncode"] == 0
        results.append(
            {
                "name": task.name,
                "summary": task.summary,
                "task_dir": str(task_dir.relative_to(ROOT)),
                "before_returncode": before["returncode"],
                "hermes": hermes,
                "after": after,
                "hidden": hidden,
                "passed": passed,
            }
        )
        print(
            f"{task.name}: {'PASS' if passed else 'FAIL'} "
            f"(hermes {hermes['duration_s']}s, tests rc={after['returncode']}, hidden rc={hidden['returncode']})",
            flush=True,
        )

    summary = {
        "run_root": str(RUN_ROOT.relative_to(ROOT)),
        "profile": PROFILE,
        "model": MODEL,
        "results": results,
        "pass_count": sum(1 for r in results if r["passed"]),
        "task_count": len(results),
    }
    (RUN_ROOT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(RUN_ROOT), "pass_count": summary["pass_count"], "task_count": len(results)}, indent=2))
    return 0 if summary["pass_count"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
