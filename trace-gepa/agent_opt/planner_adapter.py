"""PlannerAdapter — scores LLM issue-decompositions against ground-truth issues.

Targets BAG's `planDagIssues` step (src/dag-tool-loop.ts): the model reads a
high-level user request and emits
`{"issues":[{"title","body","expectedFiles","verifierCommands"}, ...]}`.

Score components (capped at 1.0):
  +0.3  baseline if JSON valid AND has `issues` list
  +0.2  if predicted issue count is within +/-2 of ground truth
  +0.3 * Jaccard overlap between predicted expectedFiles and ground truth
  +0.2  if every predicted issue has at least one verifierCommand
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch

from . import llm as _llm

_USER_TEMPLATE = (
    "Decompose the user request into a sequence of issues for a coding agent.\n\n"
    "User request:\n{user_request}\n\n"
    "Output STRICTLY as compact JSON on a single line:\n"
    '{{"issues":[{{"issueId":"task-1-...","title":"...","body":"...",'
    '"expectedFiles":["relative/path"],"verifierCommands":["bash -c \'...\'"]}}]}}'
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def _norm_path(p: str) -> str:
    if not p:
        return ""
    return re.sub(r"^/users/[^/]+/", "", p.strip().lower()).lstrip("/")


def _file_set(issues: list[dict]) -> set[str]:
    out: set[str] = set()
    for iss in issues or []:
        if not isinstance(iss, dict):
            continue
        for f in iss.get("expectedFiles") or []:
            if isinstance(f, str) and f:
                out.add(_norm_path(f))
                bn = Path(f).name.lower()
                if bn:
                    out.add(bn)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def _score(record: dict, parsed: dict | None) -> tuple[float, dict]:
    bd = {"json_valid": 0.0, "count_match": 0.0, "file_jaccard": 0.0, "verifier_coverage": 0.0}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("issues"), list):
        return 0.0, bd
    issues = parsed["issues"]
    bd["json_valid"] = 0.3
    gt = record.get("ground_truth_issues") or []
    if abs(len(issues) - len(gt)) <= 2:
        bd["count_match"] = 0.2
    bd["file_jaccard"] = round(0.3 * _jaccard(_file_set(issues), _file_set(gt)), 4)
    if issues:
        ok = sum(
            1 for iss in issues
            if isinstance(iss, dict)
            and isinstance(iss.get("verifierCommands"), list)
            and any(isinstance(v, str) and v.strip() for v in iss["verifierCommands"])
        )
        if ok == len(issues):
            bd["verifier_coverage"] = 0.2
    return min(1.0, sum(bd.values())), bd


class PlannerAdapter:
    """GEPA adapter for BAG planner-step prompts."""

    propose_new_texts = None

    def __init__(self, task_model: str = "claude-opus-4-7", max_tokens: int = 1500):
        self.task_model = task_model
        self.max_tokens = max_tokens

    def evaluate(self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        system_prompt = candidate.get("system", "")
        outputs: list[dict | None] = []
        scores: list[float] = []
        traj: list[dict] = []
        for record in batch:
            user_prompt = _USER_TEMPLATE.format(user_request=str(record.get("user_request") or "")[:1500])
            try:
                raw = _llm.chat(
                    messages=[{"role": "user", "content": user_prompt}],
                    model=self.task_model, max_tokens=self.max_tokens,
                    temperature=0.0, system=system_prompt,
                )
                err = None
            except Exception as e:
                raw, err = "", str(e)
            parsed = _parse_json(raw)
            score, bd = _score(record, parsed)
            outputs.append(parsed); scores.append(score)
            if capture_traces:
                traj.append({"input_record": record, "predicted": parsed, "score": score,
                             "breakdown": bd, "raw_output": raw, "error": err})
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traj if capture_traces else None)

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        recs: list[dict[str, Any]] = []
        for tr in eval_batch.trajectories or []:
            if not isinstance(tr, dict):
                continue
            rec = tr.get("input_record") or {}
            predicted = tr.get("predicted")
            score = tr.get("score", 0.0)
            bd = tr.get("breakdown") or {}
            gt = rec.get("ground_truth_issues") or []
            pred_issues = predicted["issues"] if isinstance(predicted, dict) and isinstance(predicted.get("issues"), list) else []
            missed = sorted(_file_set(gt) - _file_set(pred_issues))[:6]
            extra = sorted(_file_set(pred_issues) - _file_set(gt))[:6]
            parts = [f"score={score:.2f}", f"breakdown={bd}"]
            if not pred_issues and not isinstance(predicted, dict):
                parts.append("BAD: output was not valid JSON or missing 'issues' list.")
            else:
                if abs(len(pred_issues) - len(gt)) > 2:
                    parts.append(f"COUNT_MISS: predicted {len(pred_issues)} but ground truth had {len(gt)}.")
                if missed:
                    parts.append(f"MISSED_FILES: {missed}")
                if extra:
                    parts.append(f"EXTRA_FILES: {extra}")
                if pred_issues and bd.get("verifier_coverage", 0) < 0.2:
                    parts.append("VERIFIERS: not every predicted issue has a verifierCommand.")
                if score >= 0.9:
                    parts.append("GOOD: decomposition closely matches the original session's work units.")
            recs.append({
                "Inputs": {"user_request": str(rec.get("user_request") or "")[:600]},
                "Generated Outputs": predicted if predicted is not None else {},
                "Feedback": " | ".join(parts), "score": score,
            })
        out: dict[str, list[dict[str, Any]]] = {}
        for comp in components_to_update or ["system"]:
            out[comp] = list(recs)
        if "system" not in out:
            out["system"] = list(recs)
        return out
