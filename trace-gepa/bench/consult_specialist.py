"""Phase-3 specialist consultation via Codex CLI (GPT-5.5, xhigh reasoning).

Runs THREE consultations sequentially:
  1. Benchmark + verifier critique.
  2. Why xhigh hurt vs high (n=30 minibench).
  3. Synthetic-task credibility for ranking real Codex builds.

Per consultation:
  - 300s timeout.
  - subprocess: codex exec --json -c model_reasoning_effort=<effort> --model gpt-5.5 -.
  - Parse JSON-event stream from stdout, extract agent_message.text from final
    item.completed event.
  - On timeout / empty answer: drop reasoning to high and retry once.
  - On second failure: fall back to claude-opus-4-7 invoked via the Anthropic CLI
    (`claude` binary), with a prompt prefix marking it as "GPT-5.5-style critic".

Outputs:
  - bench/specialist_consultation.md
  - bench/specialist_consultation.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent  # trace-gepa/
DATA_DIR = ROOT / "data"
BENCH_DIR = ROOT / "bench"

CODEX_BIN = shutil.which("codex") or "/opt/homebrew/bin/codex"
CLAUDE_BIN = shutil.which("claude") or "/opt/homebrew/bin/claude"

OUT_MD = BENCH_DIR / "specialist_consultation.md"
OUT_JSON = BENCH_DIR / "specialist_consultation.json"

PER_CONSULT_TIMEOUT_S = 300


# ---------------------------------------------------------------------------
# Sample tasks (verbatim, trimmed) for prompt 1.
# ---------------------------------------------------------------------------

def _load_first(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    return json.loads(line)
                except Exception:
                    return None
    return None


def _trim_task_view(t: dict, ur_max: int = 700) -> dict:
    pr = t.get("prompt", {}) or {}
    user_req = pr.get("user_request") or ""
    if isinstance(user_req, str) and len(user_req) > ur_max:
        user_req = user_req[:ur_max] + " ...[truncated]"
    ctx = pr.get("context", {}) or {}
    return {
        "id": t.get("id"),
        "category": t.get("category"),
        "difficulty": t.get("difficulty"),
        "user_request_excerpt": user_req,
        "available_tools_count": len(ctx.get("available_tools", []) or []),
        "available_tools_first_8": (ctx.get("available_tools") or [])[:8],
        "expected": t.get("expected"),
        "verifier_kind": t.get("verifier_kind"),
        "verifier_spec": t.get("verifier_spec"),
        "summary": t.get("human_readable_summary"),
    }


def gather_samples() -> dict[str, dict]:
    files = {
        "tool_routing": DATA_DIR / "benchmarks" / "tool_routing.tasks.jsonl",
        "recovery": DATA_DIR / "benchmarks" / "recovery.tasks.jsonl",
        "path_grounding": DATA_DIR / "benchmarks" / "path_grounding.tasks.jsonl",
    }
    out: dict[str, dict] = {}
    for k, p in files.items():
        t = _load_first(p)
        if t is not None:
            out[k] = _trim_task_view(t)
    return out


# ---------------------------------------------------------------------------
# Codex JSON-stream invocation.
# ---------------------------------------------------------------------------

def parse_json_stream(stdout: str) -> str:
    """Extract the final agent_message.text from a codex --json event stream.

    Lines are JSON objects, one per line. We pick the last `item.completed`
    event whose `item.type == "agent_message"`, then return its `text`.
    """
    last_text = ""
    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except Exception:
            continue
        if ev.get("type") == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") == "agent_message":
                txt = item.get("text") or ""
                if txt:
                    last_text = txt
    return last_text


def codex_consult(prompt: str, *, reasoning: str, timeout: int) -> dict:
    """One codex call. Returns dict with answer, latency_s, source, exit_code, reasoning_used."""
    t0 = time.time()
    cmd = [
        CODEX_BIN, "exec",
        "-c", f'model_reasoning_effort="{reasoning}"',
        "--model", "gpt-5.5",
        "--json",
        "--skip-git-repo-check",
        "--sandbox", "read-only",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")) if e.stdout else ""
        return {
            "answer": "",
            "raw_stdout_tail": partial[-2000:] if partial else "",
            "latency_s": time.time() - t0,
            "exit_code": -1,
            "reasoning_used": reasoning,
            "source": f"codex_timeout:gpt-5.5:{reasoning}",
        }
    latency = time.time() - t0
    answer = parse_json_stream(proc.stdout or "")
    return {
        "answer": answer,
        "raw_stdout_tail": (proc.stdout or "")[-1500:] if not answer else "",
        "raw_stderr_tail": (proc.stderr or "")[-1500:] if proc.returncode != 0 else "",
        "latency_s": latency,
        "exit_code": proc.returncode,
        "reasoning_used": reasoning,
        "source": f"codex:gpt-5.5:{reasoning}",
    }


def claude_fallback(prompt: str, *, timeout: int) -> dict:
    """Claude Opus fallback, marked as GPT-5.5-style critic."""
    t0 = time.time()
    wrapped = (
        "[ROLE] You are standing in for a GPT-5.5 xhigh-reasoning specialist "
        "(Codex CLI was unreachable). Answer in the persona of a senior LM-eval "
        "engineer with the same rigor and concision a GPT-5.5 critic would apply. "
        "Do not preface with apologies. No emojis.\n\n"
        f"[ORIGINAL PROMPT]\n{prompt}"
    )
    if not Path(CLAUDE_BIN).exists() and not shutil.which("claude"):
        return {
            "answer": "[FALLBACK_UNAVAILABLE] codex failed and `claude` CLI is not on PATH.",
            "latency_s": time.time() - t0,
            "exit_code": -3,
            "reasoning_used": "n/a",
            "source": "fallback_missing",
        }
    cmd = [CLAUDE_BIN, "-p", "--model", "claude-opus-4-7"]
    try:
        proc = subprocess.run(
            cmd,
            input=wrapped,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "answer": "[FALLBACK_TIMEOUT] claude opus fallback exceeded timeout.",
            "latency_s": time.time() - t0,
            "exit_code": -1,
            "reasoning_used": "n/a",
            "source": "fallback_timeout",
        }
    return {
        "answer": (proc.stdout or "").strip(),
        "raw_stderr_tail": (proc.stderr or "")[-1000:] if proc.returncode != 0 else "",
        "latency_s": time.time() - t0,
        "exit_code": proc.returncode,
        "reasoning_used": "n/a",
        "source": "fallback:claude-opus-4-7",
    }


def run_one(label: str, prompt: str) -> dict:
    """Try xhigh, drop to high on failure, finally fall back to claude."""
    print(f"[consult] starting: {label}", flush=True)
    r = codex_consult(prompt, reasoning="xhigh", timeout=PER_CONSULT_TIMEOUT_S)
    if r.get("answer", "").strip():
        print(f"[consult] {label}: xhigh ok in {r['latency_s']:.1f}s", flush=True)
        return r
    print(f"[consult] {label}: xhigh empty/failed (exit={r.get('exit_code')}, "
          f"{r['latency_s']:.1f}s); retrying high", flush=True)
    r2 = codex_consult(prompt, reasoning="high", timeout=PER_CONSULT_TIMEOUT_S)
    if r2.get("answer", "").strip():
        r2["xhigh_attempt"] = {k: r.get(k) for k in
                               ("latency_s", "exit_code", "source", "raw_stdout_tail",
                                "raw_stderr_tail")}
        print(f"[consult] {label}: high ok in {r2['latency_s']:.1f}s", flush=True)
        return r2
    print(f"[consult] {label}: high empty/failed; falling back to claude opus", flush=True)
    r3 = claude_fallback(prompt, timeout=PER_CONSULT_TIMEOUT_S)
    r3["xhigh_attempt"] = {k: r.get(k) for k in
                           ("latency_s", "exit_code", "source", "raw_stdout_tail",
                            "raw_stderr_tail")}
    r3["high_attempt"] = {k: r2.get(k) for k in
                          ("latency_s", "exit_code", "source", "raw_stdout_tail",
                           "raw_stderr_tail")}
    return r3


# ---------------------------------------------------------------------------
# Prompts.
# ---------------------------------------------------------------------------

def build_prompt_1(samples: dict[str, dict]) -> str:
    tr = json.dumps(samples.get("tool_routing", {"note": "missing"}), indent=2)
    rc = json.dumps(samples.get("recovery", {"note": "missing"}), indent=2)
    pg = json.dumps(samples.get("path_grounding", {"note": "missing"}), indent=2)
    return f"""We built a benchmark for coding-agent action selection from real Codex/Claude-Code traces. 175 total tasks (105 trace-derived + 70 synthetic) across 7 categories: tool_routing (39), edit_safety (38), path_grounding (24), debugging (20), recovery (19), planning (19), command_synthesis (16). Difficulty: 47 easy / 48 medium / 80 hard.

Sample tasks (3 verbatim):

[tool_routing example]
{tr}

[recovery example with lesson]
{rc}

[path_grounding adversarial]
{pg}

A 30-task minibench was run on opus/haiku/gpt-5.5 (high+xhigh)/gpt-5.4. Headline: opus+seed=0.233, haiku+seed=0.200, gpt-5.5+xhigh=0.167. Three categories (command_synthesis, debugging, path_grounding) scored 0.00 across all models -- likely a verifier-DSL mismatch.

Critique as senior LM-eval engineer. Output exactly this structure:
1. Strongest threat to validity (1-2 sentences)
2. Categories to split or merge (with rationale)
3. What's missing (3 items)
4. Top 5 improvements ranked by impact-to-effort

Don't pad. Don't repeat the brief.
"""


def build_prompt_2() -> str:
    return """On a single-step tool-selection benchmark (predict {tool_name, brief_reason} given context), GPT-5.5 with `model_reasoning_effort=xhigh` scored LOWER than `high` (0.167 vs 0.200, n=30). This is contrary to the usual reasoning-quality scaling.

Why might this happen? Give 3 hypotheses ranked by likelihood. For the top hypothesis, suggest a measurable test that would falsify it.
"""


def build_prompt_3() -> str:
    return """We synthesised 70 benchmark tasks with Opus to fill underrepresented categories (edit_string_not_unique, hallucinated_skill, adversarial path_grounding). They model rare failure modes that the real corpus had n=0-2 of.

Are synthetic tasks a credible eval signal for ranking real Codex CLI builds? What's the strongest critique a reviewer would raise? How would you partition the bench so synthetic vs real numbers are reported separately?
"""


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    samples = gather_samples()

    consults = [
        ("c1_benchmark_and_verifier_critique", build_prompt_1(samples)),
        ("c2_why_xhigh_hurt_vs_high", build_prompt_2()),
        ("c3_synthetic_task_credibility", build_prompt_3()),
    ]

    results: list[dict] = []
    for label, prompt in consults:
        r = run_one(label, prompt)
        r["label"] = label
        r["prompt"] = prompt
        r["prompt_chars"] = len(prompt)
        results.append(r)

    wall = time.time() - t_start

    # JSON output.
    OUT_JSON.write_text(json.dumps({
        "wallclock_s": wall,
        "codex_calls": sum(1 for r in results
                            if r.get("source", "").startswith("codex")
                            or r.get("source", "").startswith("codex_timeout")),
        "fallback_calls": sum(1 for r in results
                               if r.get("source", "").startswith("fallback")),
        "results": results,
    }, indent=2))

    # Markdown output.
    md = [
        "# Specialist Consultation -- GPT-5.5 (Codex CLI, xhigh reasoning)",
        "",
        f"- Wallclock: {wall:.1f}s",
        f"- Total consultations: {len(results)}",
        "",
    ]
    for r in results:
        md.append(f"## {r['label']}")
        md.append("")
        md.append(f"- Source: `{r.get('source')}`")
        md.append(f"- Reasoning: `{r.get('reasoning_used')}`")
        md.append(f"- Latency: {r.get('latency_s', 0):.1f}s")
        md.append(f"- Exit code: {r.get('exit_code')}")
        md.append("")
        md.append("### Prompt")
        md.append("")
        md.append("```")
        md.append(r["prompt"].rstrip())
        md.append("```")
        md.append("")
        md.append("### Answer")
        md.append("")
        md.append(r["answer"] or "[empty]")
        md.append("")
    OUT_MD.write_text("\n".join(md))

    print(f"[consult] wrote {OUT_MD}")
    print(f"[consult] wrote {OUT_JSON}")
    print(f"[consult] wallclock={wall:.1f}s n={len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
