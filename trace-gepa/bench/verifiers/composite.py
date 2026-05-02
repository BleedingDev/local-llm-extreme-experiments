"""Tier 4: composite verifier — weighted average over a list of sub-specs."""
from __future__ import annotations

from typing import Any


def verify_composite(task: dict, predicted: Any, *, dispatch=None) -> dict:
    spec = task.get("verifier_spec") or {}
    sub_specs = spec.get("verifiers") or []
    if not sub_specs:
        return {"score": 0.0, "tier": 4, "signal": "composite_empty",
                "details": {"sub_results": []}}
    if dispatch is None:
        from . import _dispatch as dispatch  # type: ignore

    sub_results, weighted_sum, weight_total = [], 0.0, 0.0
    for sub in sub_specs:
        kind = sub.get("kind") or sub.get("verifier_kind")
        weight = float(sub.get("weight", 1.0))
        sub_task = dict(task)
        sub_task["verifier_kind"] = kind
        sub_task["verifier_spec"] = sub.get("spec", {})
        result = dispatch(kind, sub_task, predicted)
        sub_results.append({"kind": kind, "weight": weight, "result": result})
        weighted_sum += float(result.get("score", 0.0)) * weight
        weight_total += weight

    score = weighted_sum / weight_total if weight_total else 0.0
    signal = ("composite_ok" if score >= 1.0
              else ("composite_partial" if score > 0 else "composite_fail"))
    return {"score": score, "tier": 4, "signal": signal,
            "details": {"sub_results": sub_results}}
