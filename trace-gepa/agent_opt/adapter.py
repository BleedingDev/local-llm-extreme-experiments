from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.core.adapter import EvaluationBatch

from . import llm as _llm

_USER_TEMPLATE = (
    "Given the context below, decide on the next action.\n\n"
    "User request:\n{user_request}\n\n"
    "Recent assistant actions (most recent last):\n{recent_actions}\n\n"
    "Available tools: {available_tools}\n\n"
    "Output STRICTLY as compact JSON on a single line:\n"
    '{{"tool_name": "<one of available_tools or empty>", "brief_reason": "<<=20 words>"}}'
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_TOOL_NAME_RE = re.compile(
    r"\b(Read|Write|Edit|MultiEdit|NotebookEdit|Bash|Grep|Glob|LS|WebFetch|WebSearch|Task)\b"
)
_READ_CMD_RE = re.compile(r"\b(cat|head|tail|less|more|wc|ls|stat|file)\b")
_SEARCH_CMD_RE = re.compile(r"\b(rg|grep|find|fd|ack)\b")
_EXEC_CMD_RE = re.compile(r"\b(npm|bun|pnpm|yarn|python|python3|pytest|go|cargo|make|node|deno)\b")
_GENERIC_REASONS = {"", "doing the action", "doing it", "do it", "action", "tool call"}


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _extract_tool_from_text(text: str) -> str | None:
    if not text:
        return None
    m = _TOOL_NAME_RE.search(text)
    return m.group(1) if m else None


def _bash_family(inp: str) -> str | None:
    if not inp:
        return None
    # Classify by leading command token (after pipes/&&/;/sudo) to avoid
    # matching argument substrings like "head" inside "alembic upgrade head".
    tokens = re.split(r"[|&;]+|\s+", inp.strip().lower())
    tokens = [t for t in tokens if t and t != "sudo"]
    if not tokens:
        return None
    head = tokens[0]
    if _SEARCH_CMD_RE.fullmatch(head):
        return "search"
    if _EXEC_CMD_RE.fullmatch(head):
        return "exec"
    if _READ_CMD_RE.fullmatch(head):
        return "read"
    return None


def _tool_family(name: str, inp: Any = None) -> str | None:
    if not name:
        return None
    n = name.strip()
    if n in ("Read",):
        return "read"
    if n in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        return "edit"
    if n in ("Grep", "Glob"):
        return "search"
    if n == "Bash":
        inp_s = inp if isinstance(inp, str) else (json.dumps(inp) if inp else "")
        return _bash_family(inp_s)
    return None


def _compact_actions(actions: list[Any] | None) -> str:
    if not actions:
        return "(none)"
    out = []
    for a in actions[-5:]:
        if isinstance(a, dict):
            name = a.get("name") or a.get("tool_name") or a.get("kind") or "?"
            inp = a.get("input")
            inp_s = inp[:120] if isinstance(inp, str) else ("" if inp is None else json.dumps(inp)[:120])
            out.append(f"- {name}: {inp_s}")
        else:
            out.append(f"- {str(a)[:160]}")
    return "\n".join(out)


def _build_user_prompt(record: dict) -> str:
    ctx = record.get("context") or {}
    tools = ctx.get("available_tools") or []
    tools_s = ", ".join(str(t) for t in tools[:25]) if isinstance(tools, list) else str(tools)
    return _USER_TEMPLATE.format(
        user_request=str(ctx.get("user_request") or "")[:1500],
        recent_actions=_compact_actions(ctx.get("recent_actions")),
        available_tools=tools_s or "(unknown)",
    )


class TraceAdapter:
    propose_new_texts = None

    def __init__(self, task_model: str = "claude-opus-4-7", max_tokens: int = 512):
        self.task_model = task_model
        self.max_tokens = max_tokens

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        system_prompt = candidate.get("system", "")
        outputs: list[dict | None] = []
        scores: list[float] = []
        trajectories: list[dict] = []
        for record in batch:
            user_prompt = _build_user_prompt(record)
            try:
                raw = _llm.chat(
                    messages=[{"role": "user", "content": user_prompt}],
                    model=self.task_model,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    system=system_prompt,
                )
                err = None
            except Exception as e:
                raw, err = "", str(e)
            parsed = _parse_json(raw)
            score = self._score(record, parsed, raw=raw)
            outputs.append(parsed)
            scores.append(score)
            if capture_traces:
                trajectories.append(
                    {"input_record": record, "predicted": parsed, "score": score, "raw_output": raw, "error": err}
                )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    @staticmethod
    def _score(record: dict, parsed: dict | None, raw: str = "") -> float:
        observed = record.get("observed_action") or {}
        observed_name = (observed.get("name") or "").strip()
        observed_input = observed.get("input") or observed.get("arguments")
        label = (record.get("label") or "").strip()

        fallback_weight = 1.0
        chosen = ""
        chosen_input: Any = None
        reason = ""
        if isinstance(parsed, dict):
            chosen = (parsed.get("tool_name") or "").strip()
            chosen_input = parsed.get("input") or parsed.get("arguments")
            reason = (parsed.get("brief_reason") or "").strip()
        if not chosen:
            tname = _extract_tool_from_text(raw)
            if tname:
                chosen = tname
                fallback_weight = 0.5
            else:
                return 0.0

        obs_family = _tool_family(observed_name, observed_input)
        pred_family = _tool_family(chosen, chosen_input)
        same_name = bool(observed_name) and chosen == observed_name
        same_family = bool(obs_family) and obs_family == pred_family

        base = 0.0
        if label == "good":
            if same_name:
                base = 1.0
            elif same_family:
                base = 0.5
        elif label in ("bad", "user_corrected"):
            if observed_name and not same_name:
                base = 1.0 if not same_family else 0.5
        elif label == "user_confirmed":
            if same_name:
                base = 1.0
            elif same_family:
                base = 0.5

        if base > 0.0 and isinstance(parsed, dict) and reason.lower() in _GENERIC_REASONS:
            base *= 0.95
        return base * fallback_weight

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        records: list[dict[str, Any]] = []
        for traj in eval_batch.trajectories or []:
            if not isinstance(traj, dict):
                continue
            rec = traj.get("input_record") or {}
            ctx = rec.get("context") or {}
            observed = rec.get("observed_action") or {}
            label = (rec.get("label") or "").strip()
            failure = rec.get("failure_category")
            obs_name = observed.get("name") or "?"
            predicted = traj.get("predicted")
            chosen = (predicted.get("tool_name") or "").strip() if isinstance(predicted, dict) else ""
            score = traj.get("score", 0.0)
            if score >= 1.0 and label in ("bad", "user_corrected"):
                feedback = f"GOOD: avoided previously-failed action `{obs_name}` (failure: `{failure or 'unknown'}`)."
            elif score < 1.0 and label in ("bad", "user_corrected"):
                feedback = f"BAD: chose `{chosen or '<none>'}` which previously failed with `{failure or 'unknown'}`. Pick a different approach."
            elif score >= 1.0 and label == "good":
                feedback = f"GOOD: reproduced known-good action `{obs_name}`."
            elif score < 1.0 and label == "good":
                feedback = f"MISS: chose `{chosen or '<none>'}` but the working action was `{obs_name}`."
            else:
                feedback = f"score={score} label={label or 'unknown'} chosen=`{chosen or '<none>'}` observed=`{obs_name}` failure=`{failure or 'none'}`."
            records.append(
                {
                    "Inputs": {
                        "user_request": str(ctx.get("user_request") or "")[:600],
                        "recent_actions": _compact_actions(ctx.get("recent_actions")),
                        "available_tools": ctx.get("available_tools") or [],
                    },
                    "Generated Outputs": predicted if predicted is not None else {},
                    "Feedback": feedback,
                    "score": score,
                    "label": label,
                    "failure_category": failure,
                }
            )
        out: dict[str, list[dict[str, Any]]] = {}
        for comp in components_to_update or ["system"]:
            out[comp] = list(records)
        if "system" not in out:
            out["system"] = list(records)
        return out
