"""Tier 1 verifiers. Reads `pattern_or_command` (benchmark_tasks.jsonl); falls back to legacy `pattern`/`schema`."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from agent_opt.adapter import _tool_family  # type: ignore
except Exception:  # pragma: no cover
    def _tool_family(name: str, inp: Any = None) -> str | None: return None


def _as_text(p: Any) -> str:
    if isinstance(p, str):
        return p
    if p is None:
        return ""
    try:
        return json.dumps(p, sort_keys=True)
    except Exception:
        return str(p)


def _as_obj(p: Any) -> Any:
    if isinstance(p, (dict, list)):
        return p
    if isinstance(p, str):
        try: return json.loads(p)
        except Exception: return None
    return None


def _norm_ws(s: str) -> str: return re.sub(r"\s+", " ", (s or "").strip())


def _input_text(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    inp = obj.get("input")
    if isinstance(inp, str):
        return inp
    if inp is None:
        return str(obj.get("brief_reason") or "")
    try:
        return json.dumps(inp, sort_keys=True)
    except Exception:
        return str(inp)


def verify_regex(task: dict, predicted: Any) -> dict:
    """Match `pattern_or_command` (fallback `pattern`) against predicted JSON / input / brief_reason."""
    spec = task.get("verifier_spec") or {}
    pattern = spec.get("pattern_or_command") or spec.get("pattern") or ""
    cs = bool(spec.get("case_sensitive", False)) or (
        "case_insensitive" in spec and not bool(spec.get("case_insensitive")))
    flags = 0 if cs else re.IGNORECASE
    if not pattern:
        return {"score": 0.0, "tier": 1, "signal": "regex_no_pattern", "details": {}}
    obj = _as_obj(predicted)
    hs = [_as_text(predicted)]
    if isinstance(obj, dict):
        hs += [_input_text(obj), str(obj.get("brief_reason") or "")]
    try:
        ok = any(h and re.search(pattern, h, flags=flags) for h in hs)
    except re.error as e:
        return {"score": 0.0, "tier": 1, "signal": "regex_error", "details": {"error": str(e)}}
    return {"score": 1.0 if ok else 0.0, "tier": 1,
            "signal": "regex_match" if ok else "regex_miss", "details": {"pattern": pattern}}


def verify_exact_match(task: dict, predicted: Any) -> dict:
    spec = task.get("verifier_spec") or {}
    expected = spec.get("expected", task.get("expected_output", ""))
    pn, en = _norm_ws(_as_text(predicted)), _norm_ws(_as_text(expected))
    ok = pn == en
    return {"score": 1.0 if ok else 0.0, "tier": 1,
            "signal": "exact_match" if ok else "exact_miss",
            "details": {"expected": en[:120], "got": pn[:120]}}


_TYPE_MAP = {"object": dict, "array": list, "string": str, "number": (int, float),
             "integer": int, "boolean": bool, "null": type(None)}


def _check_schema(value: Any, schema: dict) -> tuple[bool, str]:
    t = schema.get("type")
    if t is not None:
        py = _TYPE_MAP.get(t)
        if py is None:
            return False, f"unknown type {t!r}"
        if t == "integer" and isinstance(value, bool):
            return False, "expected integer, got bool"
        if not isinstance(value, py):
            return False, f"expected {t}, got {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return False, "value not in enum"
    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        return False, "string too short"
    if isinstance(value, dict):
        for r in schema.get("required") or []:
            if r not in value:
                return False, f"missing required {r!r}"
        for k, sub in (schema.get("properties") or {}).items():
            if k in value and not (rv := _check_schema(value[k], sub))[0]:
                return False, f"{k}: {rv[1]}"
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return False, "array too short"
        if (items := schema.get("items")):
            for i, v in enumerate(value):
                if not (rv := _check_schema(v, items))[0]:
                    return False, f"[{i}]: {rv[1]}"
    return True, ""


# DSL for `pattern_or_command` (structural_json).
_EQ_RE = re.compile(r"\$\.([\w.]+)\s*==\s*\"([^\"]+)\"")
_NE_RE = re.compile(r"\$\.([\w.]+)\s*!=\s*null", re.IGNORECASE)
_IN_RE = re.compile(r"([\w.]+)\s+in\s+\{([^}]+)\}")
_MENTION_RE = re.compile(r"plan\s+must\s+mention\s+>=\s*(\d+)\s+of\s*:\s*(.*)", re.IGNORECASE)
_NO_REPEAT_RE = re.compile(r"must\s+not\s+repeat", re.IGNORECASE)


def _dotted_get(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _parse_dsl_clauses(text: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if not text:
        return out
    for raw in re.split(r"\s+and\s+", text, flags=re.IGNORECASE):
        s = raw.strip()
        if not s:
            continue
        if (m := _EQ_RE.search(s)):
            out.append(("eq", (m.group(1), m.group(2))))
        elif (m := _NE_RE.search(s)):
            out.append(("not_null", m.group(1)))
        elif (m := _IN_RE.search(s)):
            vals = [v.strip().strip('"').strip("'") for v in m.group(2).split(",")]
            out.append(("in", (m.group(1), [v for v in vals if v])))
        elif (m := _MENTION_RE.search(s)):
            items = [x.strip() for x in m.group(2).split(",") if x.strip()]
            out.append(("mention_n_of", (int(m.group(1)), items)))
        elif _NO_REPEAT_RE.search(s):
            out.append(("no_repeat", s))
        else:
            out.append(("unknown", s))
    return out


def _eval_clause(kind: str, payload: Any, obj: Any, task: dict) -> bool | None:
    if kind == "eq":
        return _dotted_get(obj, payload[0]) == payload[1]
    if kind == "not_null":
        return _dotted_get(obj, payload) not in (None, "")
    if kind == "in":
        return _dotted_get(obj, payload[0]) in payload[1]
    if kind == "mention_n_of":
        n, items = payload
        if not items:
            return None
        text = _as_text(obj).lower()
        return sum(1 for it in items if it.lower() in text) >= n
    if kind == "no_repeat":
        return _check_no_repeat(task, obj)
    return None


def _extract_recent_commands(task: dict) -> list[str]:
    """Pull command strings from recent_actions (handles dict + bare-string forms)
    and the failing 'Command:' line in the user_request."""
    ctx = ((task.get("prompt") or {}).get("context") or {})
    out: list[str] = []
    for a in (ctx.get("recent_actions") or []):
        if isinstance(a, dict):
            # Prefer .input, fall back to .input_excerpt (current dataset shape).
            v = a.get("input") if a.get("input") not in (None, "") else a.get("input_excerpt")
            if isinstance(v, (dict, list)):
                try:
                    v = json.dumps(v)
                except Exception:
                    v = str(v)
            if v:
                out.append(str(v))
                # Also extract a 'cmd' / 'command' value if present (JSON string).
                try:
                    parsed = json.loads(v) if isinstance(v, str) else v
                    if isinstance(parsed, dict):
                        for key in ("cmd", "command"):
                            if isinstance(parsed.get(key), str):
                                out.append(parsed[key])
                except Exception:
                    pass
        elif isinstance(a, str):
            out.append(a)
            # Bare strings often look like 'Bash: {"command":"..."}'
            m = re.search(r'"command"\s*:\s*"([^"]+)"', a)
            if m:
                out.append(m.group(1))
            m = re.search(r'"cmd"\s*:\s*"([^"]+)"', a)
            if m:
                out.append(m.group(1))
    # Also pull the explicit 'Command: ...' line that debugging prompts include.
    user_req = (task.get("prompt") or {}).get("user_request") or ""
    m = re.search(r"Command:\s*([^\n]+)", user_req)
    if m:
        out.append(m.group(1).strip().strip('"'))
    return [c for c in (s.strip() for s in out) if c]


def _check_no_repeat(task: dict, obj: Any) -> bool | None:
    cmds = _extract_recent_commands(task)
    if not cmds:
        return None
    out = _as_text(obj)
    # The 'failing' command is conventionally the most recent / explicit one.
    # Treat it as a substring check; require the predicted output not to contain
    # any meaningful chunk verbatim (>= 12 chars to avoid trivial overlap).
    return not any(len(c) >= 12 and c in out for c in cmds)


def _explicit_subchecks(spec: dict, obj: Any) -> list[tuple[str, bool]]:
    r: list[tuple[str, bool]] = []
    tn = obj.get("tool_name") if isinstance(obj, dict) else None
    text = _input_text(obj).lower()
    if "tool_name_must_equal" in spec:
        r.append(("tool_name_must_equal", tn == spec["tool_name_must_equal"]))
    if "tool_name_must_avoid" in spec:
        r.append(("tool_name_must_avoid", tn not in (spec.get("tool_name_must_avoid") or [])))
    if "tool_family_must_equal" in spec:
        gf = _tool_family(tn or "", obj.get("input") if isinstance(obj, dict) else None)
        r.append(("tool_family_must_equal", gf == spec["tool_family_must_equal"]))
    if "must_include_keywords_in_input" in spec:
        kws = spec.get("must_include_keywords_in_input") or []
        r.append(("must_include_keywords_in_input", all(k.lower() in text for k in kws)))
    if "must_avoid_keywords_in_input" in spec:
        kws = spec.get("must_avoid_keywords_in_input") or []
        r.append(("must_avoid_keywords_in_input", not any(k.lower() in text for k in kws)))
    if "input_pattern_regex" in spec:
        try:
            ok = bool(re.search(spec["input_pattern_regex"], _input_text(obj), flags=re.IGNORECASE))
        except re.error:
            ok = False
        r.append(("input_pattern_regex", ok))
    return r


def verify_structural_json(task: dict, predicted: Any) -> dict:
    """Avg of sub-checks from `pattern_or_command` DSL, rich fields, or legacy `schema`."""
    spec = task.get("verifier_spec") or {}
    obj = _as_obj(predicted)
    if obj is None:
        return {"score": 0.0, "tier": 1, "signal": "json_parse_fail",
                "details": {"raw": _as_text(predicted)[:120]}}
    if spec.get("schema"):
        ok, err = _check_schema(obj, spec["schema"])
        return {"score": 1.0 if ok else 0.0, "tier": 1,
                "signal": "schema_ok" if ok else "schema_fail",
                "details": {"error": err} if err else {}}
    pat = spec.get("pattern_or_command") or ""
    results = _explicit_subchecks(spec, obj)
    n_und = 0
    for kind, payload in _parse_dsl_clauses(pat):
        v = None if kind == "unknown" else _eval_clause(kind, payload, obj, task)
        if v is None:
            n_und += 1
        else:
            results.append((kind, bool(v)))
    if not results:
        return {"score": 0.0, "tier": 1, "signal": "no_assertions_found",
                "details": {"pat": pat[:120], "undecidable": n_und}}
    score = sum(1.0 for _, v in results if v) / len(results)
    sig = "schema_ok" if score == 1.0 else ("schema_partial" if score > 0 else "schema_fail")
    return {"score": score, "tier": 1, "signal": sig,
            "details": {"sub": [{"k": k, "ok": v} for k, v in results], "undecidable": n_und}}


def verify_tool_name_match(task: dict, predicted: Any) -> dict:
    spec = task.get("verifier_spec") or {}
    exp = (spec.get("expected_tool") or spec.get("tool_name_must_equal")
           or task.get("expected_tool") or "")
    obj = _as_obj(predicted) or {}
    got = (obj.get("tool_name") if isinstance(obj, dict) else None) or ""
    ok = bool(exp) and got == exp
    return {"score": 1.0 if ok else 0.0, "tier": 1,
            "signal": "tool_name_match" if ok else "tool_name_miss",
            "details": {"expected": exp, "got": got}}


def verify_tool_family_match(task: dict, predicted: Any) -> dict:
    spec = task.get("verifier_spec") or {}
    exp_name = (spec.get("expected_tool") or spec.get("tool_name_must_equal")
                or task.get("expected_tool") or "")
    obj = _as_obj(predicted) or {}
    got_name = (obj.get("tool_name") if isinstance(obj, dict) else None) or ""
    exp_fam = spec.get("tool_family_must_equal") or _tool_family(exp_name, spec.get("expected_input"))
    got_fam = _tool_family(got_name, obj.get("input") if isinstance(obj, dict) else None)
    ok = exp_fam is not None and exp_fam == got_fam
    return {"score": 1.0 if ok else 0.0, "tier": 1,
            "signal": "tool_family_match" if ok else "tool_family_miss",
            "details": {"expected_family": exp_fam, "got_family": got_fam}}
