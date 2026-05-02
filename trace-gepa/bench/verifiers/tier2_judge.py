"""Tier 2: LM-as-judge. Chat function is injected for testability."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_JUDGE_SYSTEM = (
    "You are a strict grader. Score the predicted output against the task. "
    "Reply with STRICTLY one JSON line: "
    '{"score": <0.0..1.0>, "rationale": "<one short sentence>"}'
)
_JUDGE_USER = (
    "Task description:\n{task_desc}\n\nRubric criteria (optional):\n{rubric}\n\n"
    "Predicted output:\n{predicted}\n\nReturn ONE-LINE JSON only."
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _default_chat(messages, model, max_tokens, temperature, system):  # pragma: no cover
    from agent_opt import llm as _llm
    return _llm.chat(messages=messages, model=model, max_tokens=max_tokens,
                     temperature=temperature, system=system)


def _parse_judge(raw: str) -> dict:
    if not raw:
        return {"score": 0.0, "rationale": "empty judge reply"}
    try:
        obj = json.loads(raw)
    except Exception:
        m = _JSON_RE.search(raw)
        if not m:
            return {"score": 0.0, "rationale": "unparseable judge reply"}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {"score": 0.0, "rationale": "unparseable judge reply"}
    try:
        score_f = float(obj.get("score", 0.0))
    except Exception:
        score_f = 0.0
    score_f = max(0.0, min(1.0, score_f))
    return {"score": score_f, "rationale": str(obj.get("rationale", ""))[:240]}


def verify_lm_judge(task: dict, predicted: Any, *, chat_fn: Callable | None = None,
                    model: str = "claude-opus-4-7") -> dict:
    spec = task.get("verifier_spec") or {}
    rubric = spec.get("rubric") or task.get("rubric") or "Match intent and correctness."
    task_desc = (task.get("description") or task.get("user_request")
                 or (task.get("context") or {}).get("user_request") or "")
    pred_text = predicted if isinstance(predicted, str) else json.dumps(predicted, sort_keys=True)
    user = _JUDGE_USER.format(task_desc=str(task_desc)[:1500],
                              rubric=str(rubric)[:600], predicted=str(pred_text)[:1500])
    fn = chat_fn or _default_chat
    try:
        raw = fn(messages=[{"role": "user", "content": user}], model=model,
                 max_tokens=128, temperature=0.0, system=_JUDGE_SYSTEM)
    except Exception as e:
        return {"score": 0.0, "tier": 2, "signal": "judge_error",
                "details": {"error": str(e)[:200]}}
    parsed = _parse_judge(raw or "")
    return {"score": parsed["score"], "tier": 2, "signal": "judge_score",
            "details": {"rationale": parsed["rationale"], "raw": (raw or "")[:240]}}
