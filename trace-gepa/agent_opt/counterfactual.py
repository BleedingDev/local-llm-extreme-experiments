"""Counterfactual annotator (Proposal K).

Given a `bad`/`user_corrected` record from `data/dataset.jsonl`, ask Opus 4.7
what action a senior engineer would have taken. Output schema:

    {record_id, observed_action, counterfactual_action, rationale,
     delta_kind, confidence}

The annotator preprends a SHARED preamble (with cache_control) so Anthropic
prompt caching amortises schema/instructions across the corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import llm as _llm
from .adapter import _compact_actions, _parse_json

_VALID_DELTA_KINDS = {"tool_swap", "input_fix", "abort", "decompose", "verify_first"}

_PREAMBLE = (
    "You are a senior staff engineer reviewing a Claude Code agent session in"
    " which the agent took an action that FAILED or was overruled by the user."
    " Your job: given only what was knowable at decision time, output the"
    " action a senior engineer would have taken instead.\n\n"
    "Output STRICT JSON, single object, no prose, no markdown fences. Schema:\n"
    "{\n"
    '  "counterfactual_action": {"name": "<tool name OR empty if abort>",'
    ' "input": <object or string, may be null>},\n'
    '  "rationale": "<<=40 words, why this action is better>",\n'
    '  "delta_kind": "tool_swap" | "input_fix" | "abort" | "decompose" |'
    ' "verify_first",\n'
    '  "confidence": <float 0..1>\n'
    "}\n\n"
    "delta_kind guide:\n"
    "- tool_swap: different tool would have worked (e.g. Read instead of Bash cat)\n"
    "- input_fix: same tool, fixed arguments (e.g. add --fix flag, correct path)\n"
    "- abort: no good action existed; agent should have asked the user or stopped\n"
    "- decompose: action was too coarse; should split into smaller steps first\n"
    "- verify_first: agent should have inspected state (Read/LS/Grep) before acting\n\n"
    "Do NOT output 'I cannot determine' style refusals. If the trace is thin,"
    " still pick the best plausible action and lower the confidence."
)


def _truncate(s: str | None, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_observed(observed: dict | None) -> str:
    if not isinstance(observed, dict):
        return "(unknown)"
    name = observed.get("name") or observed.get("kind") or "?"
    inp = observed.get("input")
    if isinstance(inp, str):
        try:
            inp = json.loads(inp)
        except Exception:
            pass
    inp_s = _truncate(json.dumps(inp, ensure_ascii=False) if inp is not None else "", 600)
    res = _truncate(str(observed.get("result_excerpt") or ""), 500)
    err = bool(observed.get("result_is_error"))
    return f"name={name}\ninput={inp_s}\nresult_is_error={err}\nresult_excerpt={res}"


def _build_user_msg(record: dict) -> str:
    ctx = record.get("context") or {}
    user_request = _truncate(str(ctx.get("user_request") or ""), 1500)
    recent = _compact_actions(ctx.get("recent_actions"))
    observed = _format_observed(record.get("observed_action"))
    failure = record.get("failure_category") or "(none)"
    next_user = _truncate(str(record.get("next_user_message") or ""), 600)
    tools = ctx.get("available_tools") or []
    tools_s = ", ".join(str(t) for t in tools[:30]) if isinstance(tools, list) else str(tools)
    return (
        f"<user_request>\n{user_request}\n</user_request>\n\n"
        f"<recent_actions>\n{recent}\n</recent_actions>\n\n"
        f"<available_tools>{tools_s}</available_tools>\n\n"
        f"<observed_action_that_failed>\n{observed}\n</observed_action_that_failed>\n\n"
        f"<failure_category>{failure}</failure_category>\n\n"
        f"<next_user_message>\n{next_user}\n</next_user_message>\n\n"
        "Return the JSON object now."
    )


def _is_copout(rationale: str | None) -> bool:
    if not rationale:
        return True
    low = rationale.lower()
    triggers = (
        "cannot determine",
        "not enough context",
        "without more context",
        "insufficient information",
        "unable to determine",
    )
    return any(t in low for t in triggers)


def _validate(parsed: dict | None) -> dict | None:
    if not isinstance(parsed, dict):
        return None
    cfa = parsed.get("counterfactual_action")
    if not isinstance(cfa, dict):
        return None
    delta = parsed.get("delta_kind")
    if delta not in _VALID_DELTA_KINDS:
        return None
    rationale = parsed.get("rationale") or ""
    if _is_copout(rationale):
        return None
    try:
        conf = float(parsed.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    return {
        "counterfactual_action": cfa,
        "rationale": str(rationale)[:400],
        "delta_kind": delta,
        "confidence": max(0.0, min(1.0, conf)),
    }


def _chat_with_cache(user_msg: str, model: str, max_tokens: int) -> str:
    """Use Anthropic prompt caching by sending the preamble as a cached system block."""
    client = _llm._client()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": _PREAMBLE, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_msg}],
    }
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.messages.create(**kwargs)
            parts = []
            for block in resp.content:
                t = getattr(block, "text", None)
                if t:
                    parts.append(t)
            return "".join(parts)
        except Exception as e:
            last_err = e
            msg = str(e)
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                continue
            if attempt < 2:
                time.sleep(1.0 + attempt)
                continue
            raise
    raise last_err  # unreachable


def annotate(record: dict, model: str = "claude-opus-4-7", max_tokens: int = 600) -> dict | None:
    """Annotate a single record. Returns None on parse failure or API error."""
    rid = record.get("id") or record.get("record_id") or "?"
    observed = record.get("observed_action") or {}
    user_msg = _build_user_msg(record)
    try:
        raw = _chat_with_cache(user_msg, model=model, max_tokens=max_tokens)
    except Exception:
        return None
    parsed = _parse_json(raw)
    valid = _validate(parsed)
    if valid is None:
        return None
    return {
        "record_id": rid,
        "observed_action": {
            "name": observed.get("name") or observed.get("kind"),
            "input": observed.get("input"),
        },
        "counterfactual_action": valid["counterfactual_action"],
        "rationale": valid["rationale"],
        "delta_kind": valid["delta_kind"],
        "confidence": valid["confidence"],
    }


# ---------------------------------------------------------------------------
# CLI: smoke + full pass
# ---------------------------------------------------------------------------

_REPO = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa")
_DATASET = _REPO / "data" / "dataset.jsonl"


def _load_targets() -> list[dict]:
    out = []
    with _DATASET.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("label") in ("bad", "user_corrected"):
                out.append(r)
    return out


def _run_batch(records: list[dict], out_path: Path, model: str, workers: int = 8) -> dict:
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(annotate, r, model): r for r in records}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                ann = fut.result()
            except Exception:
                ann = None
            if ann is None:
                failures += 1
            else:
                results.append(ann)
            if i % 25 == 0 or i == len(futs):
                print(f"  [{i}/{len(futs)}] ok={len(results)} fail={failures}", flush=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "total": len(records),
        "ok": len(results),
        "fail": failures,
        "wall_s": round(time.time() - t0, 1),
        "out": str(out_path),
    }


def _print_samples(results: list[dict], k: int = 10) -> None:
    sample = random.sample(results, min(k, len(results)))
    for r in sample:
        obs = r.get("observed_action", {})
        cf = r.get("counterfactual_action", {})
        obs_in = json.dumps(obs.get("input"), ensure_ascii=False) if obs.get("input") else ""
        cf_in = json.dumps(cf.get("input"), ensure_ascii=False) if cf.get("input") else ""
        print(f"  - id={r['record_id']} delta={r['delta_kind']} conf={r['confidence']:.2f}")
        print(f"      OBS  {obs.get('name')}: {_truncate(obs_in, 200)}")
        print(f"      CF   {cf.get('name')}: {_truncate(cf_in, 200)}")
        print(f"      WHY  {_truncate(r.get('rationale',''), 220)}")


def _aggregate(path: Path) -> dict:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    deltas = Counter(r.get("delta_kind") for r in rows)
    confs = [r.get("confidence", 0.0) for r in rows if isinstance(r.get("confidence"), (int, float))]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    aborts = deltas.get("abort", 0)
    return {"n": len(rows), "delta_kind": dict(deltas), "mean_confidence": round(mean_conf, 3), "aborts": aborts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    targets = _load_targets()
    print(f"Loaded {len(targets)} bad+user_corrected records.", flush=True)

    if args.mode == "smoke":
        recs = targets[: args.limit]
        out = Path(args.out) if args.out else _REPO / "data" / "counterfactuals_smoke.jsonl"
        print(f"SMOKE: annotating {len(recs)} records -> {out}", flush=True)
        meta = _run_batch(recs, out, args.model, workers=args.workers)
        agg = _aggregate(out)
        print(json.dumps({"smoke": meta, "agg": agg}, indent=2))
        results = [json.loads(l) for l in out.open()]
        print("\nSample (10 random):")
        _print_samples(results, 10)
        ok_pct = (agg["n"] / max(1, meta["total"])) * 100.0
        print(f"\nParseable+plausible: {agg['n']}/{meta['total']} = {ok_pct:.1f}%")
        verdict = "PROCEED" if ok_pct >= 80.0 else "STOP"
        print(f"VERDICT: {verdict}")
        return 0 if verdict == "PROCEED" else 2

    # full
    out = Path(args.out) if args.out else _REPO / "data" / "counterfactuals.jsonl"
    print(f"FULL: annotating {len(targets)} records -> {out}", flush=True)
    meta = _run_batch(targets, out, args.model, workers=args.workers)
    agg = _aggregate(out)
    summary = {"meta": meta, "agg": agg}
    print(json.dumps(summary, indent=2))
    summary_md = _REPO / "data" / "counterfactuals_summary.md"
    with summary_md.open("w") as f:
        f.write("# Counterfactual Annotations Summary\n\n")
        f.write(f"- records (input): {meta['total']}\n")
        f.write(f"- records (output, valid): {agg['n']}\n")
        f.write(f"- failed/skipped: {meta['fail']}\n")
        f.write(f"- wallclock seconds: {meta['wall_s']}\n")
        f.write(f"- mean confidence: {agg['mean_confidence']}\n")
        f.write(f"- abort verdicts: {agg['aborts']}\n\n")
        f.write("## delta_kind distribution\n\n")
        for k, v in sorted(agg["delta_kind"].items(), key=lambda kv: -kv[1]):
            f.write(f"- {k}: {v}\n")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
