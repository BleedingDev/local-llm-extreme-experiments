"""Session-replay LM-judge oracle (proposal: session_replay_oracle).

The judge sees only the user's request, the agent's predicted action, and the
user's actual follow-up message. It does NOT see the original observed_action
or the dataset label, so it can be used to score arbitrary candidate actions.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt.llm import chat  # noqa: E402

_JUDGE_TEMPLATE = """A user gave this request:
<<<USER_REQUEST
{user_request}
USER_REQUEST

The agent did this:
- tool: {tool_name}
- brief reason: {brief_reason}

The user then responded:
<<<USER_FOLLOWUP
{next_user_message}
USER_FOLLOWUP

Based ONLY on the user's actual follow-up, did the agent's action satisfy the user?
Score 0 (rejected/wrong), 0.5 (partial/ambiguous), or 1 (accepted/correct).
Output ONLY a single number: 0, 0.5, or 1."""


def _brief_from_observed(observed: dict) -> dict:
    """Render an `observed_action` dict into a {tool_name, brief_reason} pair."""
    name = observed.get("name") or observed.get("tool_name") or "unknown"
    raw_input = observed.get("input")
    if isinstance(raw_input, str):
        try:
            raw_input = json.loads(raw_input)
        except Exception:
            pass
    if isinstance(raw_input, dict):
        kv = ", ".join(f"{k}={str(v)[:80]}" for k, v in list(raw_input.items())[:4])
    else:
        kv = str(raw_input)[:160]
    return {"tool_name": name, "brief_reason": f"invoke {name} with {kv}"}


def _parse_score(raw: str) -> float | None:
    """Extract the first 0/0.5/1 in the model's output."""
    if raw is None:
        return None
    match = re.search(r"\b(0\.5|0|1(?:\.0+)?)\b", raw.strip())
    if not match:
        return None
    val = float(match.group(1))
    if val not in (0.0, 0.5, 1.0):
        return None
    return val


def score_action_via_followup(
    record: dict,
    predicted_action: dict,
    judge_model: str = "claude-opus-4-7",
) -> dict:
    """Score a predicted action against the user's actual follow-up.

    `record` must contain `context.user_request` and `next_user_message`.
    `predicted_action` is `{tool_name, brief_reason}` (no other fields read).
    """
    ctx = record.get("context") or {}
    user_request = ctx.get("user_request") or ""
    followup = record.get("next_user_message") or ""
    if not followup:
        return {"score": None, "raw": "", "latency": 0.0, "skipped": "no_followup"}

    prompt = _JUDGE_TEMPLATE.format(
        user_request=str(user_request)[:4000],
        tool_name=predicted_action.get("tool_name", "unknown"),
        brief_reason=predicted_action.get("brief_reason", ""),
        next_user_message=str(followup)[:2000],
    )

    t0 = time.time()
    try:
        raw = chat(
            messages=[{"role": "user", "content": prompt}],
            model=judge_model,
            max_tokens=8,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return {"score": None, "raw": f"<error: {exc}>", "latency": time.time() - t0}
    latency = time.time() - t0
    return {"score": _parse_score(raw), "raw": (raw or "").strip(), "latency": latency}


def score_observed(record: dict, **kw: Any) -> dict:
    """Convenience: score the record's own observed_action via the judge."""
    return score_action_via_followup(record, _brief_from_observed(record["observed_action"]), **kw)


def synthesize_ideal_answer(record: dict) -> dict:
    """Heuristic ideal answer derived from `ideal_action_hint` (round-9 reverse_validator).

    The dataset stores `ideal_action_hint` as a free-form string. We pair it
    with the observed action's tool name (the closest proxy for an "ideal"
    action when the hint doesn't carry structured tool info).
    """
    hint = record.get("ideal_action_hint")
    if isinstance(hint, dict):
        tool_name = hint.get("tool_name") or hint.get("name") or "Read"
        rationale = hint.get("rationale") or hint.get("reason") or "fulfil the user's request"
    else:
        observed = record.get("observed_action") or {}
        tool_name = observed.get("name") or "Read"
        rationale = (hint or "fulfil the user's request").strip()[:240]
    return {
        "tool_name": tool_name,
        "brief_reason": f"{rationale} (invoking {tool_name} as required)",
    }
