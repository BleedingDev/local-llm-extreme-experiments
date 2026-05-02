"""Inference-time ensemble over multiple optimised system prompts.

Runs K candidate prompts in parallel against the same input, then aggregates
either by majority vote on `tool_name` or by asking a judge LM which output is
most likely correct. No GEPA, no training — purely test-time.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import llm as _llm
from .adapter import _build_user_prompt, _parse_json

_JUDGE_INDEX_RE = re.compile(r"\b(\d+)\b")


def _format_output(parsed: Any, raw: str) -> str:
    if isinstance(parsed, dict):
        try:
            return json.dumps(parsed, separators=(",", ":"))
        except Exception:
            pass
    return (raw or "").strip()[:400]


_JUDGE_SYSTEM = (
    "You are a strict judge. Given a coding-agent decision context and several "
    "candidate JSON responses, choose the single candidate index whose tool_name "
    "is most likely the correct next action. Output ONLY the integer index."
)


def _judge_prompt(user_prompt: str, candidates: list[tuple[str, str]]) -> str:
    lines = ["Context (the user prompt the candidates each saw):", "", user_prompt, "", "Candidates:"]
    for i, (_name, out) in enumerate(candidates):
        lines.append(f"[{i}] {out}")
    lines += [
        "",
        "Respond with the integer index (0-based) of the best candidate. Output only the integer.",
    ]
    return "\n".join(lines)


class EnsemblePredictor:
    def __init__(
        self,
        prompts: list[tuple[str, str]],
        task_model: str = "claude-opus-4-7",
        judge_model: str = "claude-opus-4-7",
        aggregator: str = "judge",
        max_tokens: int = 512,
    ) -> None:
        if not prompts:
            raise ValueError("prompts must be non-empty")
        if aggregator not in ("vote", "judge"):
            raise ValueError(f"aggregator must be 'vote' or 'judge', got {aggregator!r}")
        self.prompts = prompts
        self.task_model = task_model
        self.judge_model = judge_model
        self.aggregator = aggregator
        self.max_tokens = max_tokens

    def _call_one(self, system_prompt: str, user_prompt: str) -> tuple[dict | None, str]:
        try:
            raw = _llm.chat(
                messages=[{"role": "user", "content": user_prompt}],
                model=self.task_model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                system=system_prompt,
            )
        except Exception:
            raw = ""
        return _parse_json(raw), raw

    def _judge(self, user_prompt: str, formatted: list[tuple[str, str]]) -> int:
        try:
            judge_raw = _llm.chat(
                messages=[{"role": "user", "content": _judge_prompt(user_prompt, formatted)}],
                model=self.judge_model,
                max_tokens=8,
                temperature=0.0,
                system=_JUDGE_SYSTEM,
            )
        except Exception:
            return 0
        m = _JUDGE_INDEX_RE.search(judge_raw or "")
        if not m:
            return 0
        idx = int(m.group(1))
        return idx if 0 <= idx < len(formatted) else 0

    def predict(self, record: dict) -> dict:
        user_prompt = _build_user_prompt(record)
        with ThreadPoolExecutor(max_workers=max(1, len(self.prompts))) as ex:
            results = list(
                ex.map(lambda sp: self._call_one(sp, user_prompt), [p for _, p in self.prompts])
            )
        all_outputs = [
            {"name": name, "parsed": parsed, "raw": raw}
            for (name, _), (parsed, raw) in zip(self.prompts, results)
        ]
        formatted = [
            (self.prompts[i][0], _format_output(parsed, raw))
            for i, (parsed, raw) in enumerate(results)
        ]

        judge_idx: int | None = None
        if self.aggregator == "vote":
            tools = []
            for parsed, _raw in results:
                if isinstance(parsed, dict):
                    tn = (parsed.get("tool_name") or "").strip()
                    if tn:
                        tools.append(tn)
            if not tools:
                chosen_idx = 0
            else:
                counts = Counter(tools)
                top_count = max(counts.values())
                winners = sorted(t for t, c in counts.items() if c == top_count)
                winner = winners[0]
                chosen_idx = next(
                    (i for i, (parsed, _r) in enumerate(results)
                     if isinstance(parsed, dict)
                     and (parsed.get("tool_name") or "").strip() == winner),
                    0,
                )
        else:
            judge_idx = self._judge(user_prompt, formatted)
            chosen_idx = judge_idx

        chosen_name, _ = self.prompts[chosen_idx]
        chosen_parsed, _chosen_raw = results[chosen_idx]
        return {
            "chosen_name": chosen_name,
            "chosen_index": chosen_idx,
            "chosen_output": chosen_parsed,
            "all_outputs": all_outputs,
            "judge_score": judge_idx,
        }
