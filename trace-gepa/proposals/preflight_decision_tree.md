# Preflight Decision Tree — Deterministic Pre-Tool-Call Veto

**Author:** Brainstorm Round-2 Member #J
**Date:** 2026-05-01
**Owns:** `trace-gepa/proposals/preflight_decision_tree.md`

---

## TLDR

- **Hypothesis:** ≥80% of failures in `dataset_v2.jsonl` reduce to ≤10 deterministic predicates that can be evaluated in <1ms with zero LM cost.
- **Mechanism:** A pure-Python `PreflightChecker` runs every predicate before tool dispatch; vetoes return a structured error to the agent, which retries with a corrected action.
- **Wiring:** Hook into BAG's tool dispatcher (`pre_tool_use` event) and ship as an MCP middleware so other harnesses benefit. Every veto is logged with `{action, predicate, message, ts}` for offline mining of new predicates.
- **Why novel:** prior proposals in this round bolt LM-based critics or pattern miners on top; this is the cheapest possible interception layer — a decision tree of `if action.tool == "Edit": ...` branches, validated empirically against our labelled set.

## 5 Concrete Predicates

```python
def check_cmd_exists(action, ctx):  # cmd_not_found_127
    bin_ = first_token(action.get("command", ""))
    return (shutil.which(bin_) is not None, f"binary {bin_!r} not on PATH")

def check_edit_unique(action, ctx):  # edit_string_not_unique
    src = ctx.file_cache.get(action["file_path"], "")
    n = src.count(action["old_string"])
    return (n == 1, f"old_string occurs {n}x in file (need exactly 1)")

def check_file_was_read(action, ctx):  # edit_file_not_read
    return (action["file_path"] in ctx.read_paths_this_session,
            "Edit requires prior Read of file")

def check_path_exists(action, ctx):  # hallucinated_path
    p = action.get("file_path") or action.get("path")
    return (p is None or os.path.exists(p), f"path does not exist: {p}")

def check_skill_listed(action, ctx):  # hallucinated_skill
    return (action["skill"] in ctx.skill_listing,
            f"skill {action['skill']!r} not in available skills")
```

(Plus: `check_arg_size_kb` for `bash_timeout_141`, `check_parallel_fanout <= 8` for `cancelled_parallel_batch`, `check_glob_scope` for unbounded `find /`, `check_url_allowlist` for WebFetch, `check_write_overwrite_safe` for unread overwrites.)

## Aggregator

```python
class PreflightChecker:
    def check(self, action) -> Result:
        blocked, warns = [], []
        for pred in PREDICATES[action.tool]:
            ok, msg = pred(action, self.ctx)
            (blocked if pred.severity == "block" else warns).append(msg) if not ok else None
        return Result(passed=not blocked, blocked_by=blocked, warnings=warns)
```

## Implementation Outline / Effort

1. Day 1: enumerate predicates from `failure_category` taxonomy; stub each.
2. Day 2: replay `dataset_v2.jsonl` (~30K) computing `(predicate, label)` confusion matrix; tune until **0 false positives on `good`**.
3. Day 3: BAG `pre_tool_use` hook + MCP `preflight.check` tool; structured veto-log to `~/.claude/preflight.jsonl`.
4. Day 4: shadow mode (warn-only) for one week, then enforce.

**Effort:** ~2 engineer-days for v1, +3 days for empirical tuning. Total ≈1 week.

## Self-Critique

The decision tree is brittle to schema drift (new tools, renamed args) and cannot catch *semantic* errors (right syntax, wrong intent), so it must be paired with telemetry-driven predicate discovery to remain useful.
