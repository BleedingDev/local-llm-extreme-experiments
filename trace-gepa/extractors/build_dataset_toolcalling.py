"""
Build a tool-calling-focused dataset from trace records.

Reads (read-only):
  - trace-gepa/data/dataset.jsonl   (v1)
  - trace-gepa/data/dataset_v2.jsonl (v2)

Writes:
  - trace-gepa/data/dataset_toolcalling.jsonl
  - trace-gepa/data/splits_toolcalling.json

Goal: highest-signal slice of action-selection trace records, optimised for
training/optimising "next-tool-decision". See spec in agent brief.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------- paths ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC_V1 = DATA / "dataset.jsonl"
SRC_V2 = DATA / "dataset_v2.jsonl"
OUT_JSONL = DATA / "dataset_toolcalling.jsonl"
OUT_SPLITS = DATA / "splits_toolcalling.json"

# ---------- spec constants ------------------------------------------------

CANONICAL_TOOLS = {
    "Read", "Write", "Edit", "MultiEdit", "NotebookEdit",
    "Bash", "Grep", "Glob", "LS", "WebFetch", "WebSearch",
}

WELL_KNOWN_FAILURE_CATEGORIES = {
    "bash_exit_nonzero",
    "hallucinated_path",
    "retry_loop",
    "cmd_not_found_127",
}

POSITIVE_CONFIRM_TOKENS = (
    "thanks", "thank you", "great", "perfect", "looks good", "lgtm",
    "nice", "awesome", "ok cool", "yes", "good", "go ahead", "continue",
    "proceed", "do it",
)

MIN_USER_REQUEST_LEN = 20
MIN_RECENT_ACTIONS = 1
MIN_AVAILABLE_TOOLS = 3
TARGET_MIN = 2000
TARGET_MAX = 5000
BASH_DOMINANCE_THRESHOLD = 0.70
BASH_TARGET_AFTER_DOWNSAMPLE = 0.60
SEED = 1337

# ---------- io helpers ----------------------------------------------------


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_dedup_union() -> list[dict]:
    """Load v1 + v2 and dedup by id. v2 wins on collision (newer extraction)."""
    by_id: dict[str, dict] = {}
    for r in iter_jsonl(SRC_V1):
        by_id[r["id"]] = r
    for r in iter_jsonl(SRC_V2):
        by_id[r["id"]] = r
    return list(by_id.values())


# ---------- filter --------------------------------------------------------


def edit_excerpt_has_old_string_ambiguity(excerpt: str | None) -> bool:
    if not excerpt:
        return False
    s = excerpt.lower()
    return (
        "old_string" in s
        and (
            "not unique" in s
            or "no match" in s
            or "did not match" in s
            or "must be unique" in s
            or "found 0" in s
            or "ambig" in s
        )
    )


def passes_quality_filter(rec: dict) -> bool:
    oa = rec.get("observed_action") or {}
    if oa.get("kind") != "tool_use":
        return False

    name = oa.get("name")
    if name not in CANONICAL_TOOLS:
        return False

    ctx = rec.get("context") or {}

    user_request = ctx.get("user_request") or ""
    if not isinstance(user_request, str) or len(user_request) < MIN_USER_REQUEST_LEN:
        return False

    recent_actions = ctx.get("recent_actions") or []
    if len(recent_actions) < MIN_RECENT_ACTIONS:
        return False

    available_tools = ctx.get("available_tools") or []
    if len(available_tools) < MIN_AVAILABLE_TOOLS:
        return False

    if name == "Bash":
        cmd = oa.get("input")
        # Inputs may be JSON-stringified or plain strings; treat both.
        if not isinstance(cmd, str) or not cmd.strip():
            return False
        # If it parses to JSON object, ensure it has a non-empty `command`.
        try:
            parsed = json.loads(cmd)
            if isinstance(parsed, dict):
                command = parsed.get("command")
                if not isinstance(command, str) or not command.strip():
                    return False
        except (json.JSONDecodeError, TypeError):
            # Plain string command — already validated non-empty above.
            pass

    if name == "Edit":
        if edit_excerpt_has_old_string_ambiguity(oa.get("result_excerpt")):
            return False

    label = rec.get("label")

    # Negative-signal inclusion gates
    if label == "user_corrected":
        if not (rec.get("ideal_action_hint") or "").strip():
            return False
    elif label == "bad":
        fc = rec.get("failure_category")
        if fc not in WELL_KNOWN_FAILURE_CATEGORIES:
            return False
    elif label not in {"good", "user_confirmed"}:
        # Unknown / missing label — exclude conservatively.
        return False

    return True


# ---------- scoring -------------------------------------------------------


def has_positive_confirmation(next_user_message: str | None) -> bool:
    if not next_user_message:
        return True  # empty / missing counts as "no complaint"
    s = next_user_message.lower()
    return any(tok in s for tok in POSITIVE_CONFIRM_TOKENS)


def quality_score(rec: dict) -> float:
    score = 0.0
    label = rec.get("label")
    if label == "good":
        score += 0.4
    if label == "user_confirmed":
        score += 0.4
    fc = rec.get("failure_category")
    ideal = (rec.get("ideal_action_hint") or "").strip()
    if fc in WELL_KNOWN_FAILURE_CATEGORIES and ideal:
        score += 0.2

    ctx = rec.get("context") or {}
    if len(ctx.get("recent_actions") or []) >= 3:
        score += 0.1

    ur = ctx.get("user_request") or ""
    if 50 <= len(ur) <= 1000:
        score += 0.1

    # Soft bonus for positive next_user_message confirmation
    if has_positive_confirmation(rec.get("next_user_message")):
        score += 0.05

    return min(score, 1.0)


# ---------- rebalance -----------------------------------------------------


def rebalance_by_tool(
    scored: list[tuple[float, dict]],
    target_min: int,
    target_max: int,
) -> list[dict]:
    """Stratified per-tool selection.

    Strategy:
      1) Bucket by tool name.
      2) If Bash > BASH_DOMINANCE_THRESHOLD * total selected, downsample Bash to
         BASH_TARGET_AFTER_DOWNSAMPLE proportion.
      3) Within each bucket, sort by quality_score descending, take top N where
         N is computed so total lands in [target_min, target_max].
    """
    by_tool: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for s, r in scored:
        by_tool[r["observed_action"]["name"]].append((s, r))

    for t in by_tool:
        by_tool[t].sort(key=lambda x: x[0], reverse=True)

    # First-pass naive cap: include all that pass filter, up to TARGET_MAX.
    total = sum(len(v) for v in by_tool.values())
    if total <= target_max:
        keep = {t: list(v) for t, v in by_tool.items()}
    else:
        # Proportional decimation, top-quality first.
        scale = target_max / total
        keep = {
            t: v[: max(1, math.ceil(len(v) * scale))] for t, v in by_tool.items()
        }

    # Bash dominance check
    total_kept = sum(len(v) for v in keep.values())
    bash_n = len(keep.get("Bash", []))
    if total_kept and bash_n / total_kept > BASH_DOMINANCE_THRESHOLD:
        non_bash_kept = total_kept - bash_n
        # Solve: bash_new / (bash_new + non_bash_kept) == BASH_TARGET_AFTER_DOWNSAMPLE
        # bash_new = ratio * (bash_new + non_bash_kept)
        # bash_new * (1 - ratio) = ratio * non_bash_kept
        ratio = BASH_TARGET_AFTER_DOWNSAMPLE
        if non_bash_kept > 0:
            bash_new = int(round((ratio * non_bash_kept) / (1 - ratio)))
            bash_new = max(1, min(bash_new, bash_n))
            keep["Bash"] = keep["Bash"][:bash_new]

    # If we ended up below target_min, top up by re-adding next-best from any
    # tool that still has reserve.
    flat = [(s, r) for v in keep.values() for s, r in v]
    if len(flat) < target_min:
        # Build a pool of leftovers, sorted by score desc.
        kept_ids = {r["id"] for _, r in flat}
        leftovers = []
        for t, v in by_tool.items():
            for s, r in v:
                if r["id"] not in kept_ids:
                    leftovers.append((s, r))
        leftovers.sort(key=lambda x: x[0], reverse=True)
        need = target_min - len(flat)
        flat.extend(leftovers[:need])

    # Final emit (sorted globally by score desc for deterministic output)
    flat.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in flat]


# ---------- splits --------------------------------------------------------


def stratified_splits(
    records: list[dict],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = SEED,
) -> dict[str, list[str]]:
    """Stratified split by (tool_name, label). Returns {split: [id, ...]}."""
    assert abs(train + val + test - 1.0) < 1e-6
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in records:
        key = (r["observed_action"]["name"], r.get("label") or "unknown")
        by_stratum[key].append(r["id"])

    out = {"train": [], "val": [], "test": []}
    for key, ids in by_stratum.items():
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        # Anything left goes to test.
        n_test = n - n_train - n_val
        if n_test < 0:
            # Rounding edge case — borrow from train.
            n_train += n_test
            n_test = 0
        out["train"].extend(ids[:n_train])
        out["val"].extend(ids[n_train : n_train + n_val])
        out["test"].extend(ids[n_train + n_val :])

    for split in out:
        rng.shuffle(out[split])
    return out


# ---------- main ----------------------------------------------------------


def main() -> None:
    print(f"[load] {SRC_V1.name} + {SRC_V2.name}")
    union = load_dedup_union()
    print(f"[load] union (deduped by id): {len(union):,}")

    filtered: list[dict] = [r for r in union if passes_quality_filter(r)]
    print(f"[filter] passing quality bar: {len(filtered):,}")

    scored: list[tuple[float, dict]] = []
    for r in filtered:
        s = quality_score(r)
        r["quality_score"] = round(s, 4)
        scored.append((s, r))

    selected = rebalance_by_tool(scored, TARGET_MIN, TARGET_MAX)
    print(f"[rebalance] selected: {len(selected):,}")

    # Re-sort by score desc (stable) for output
    selected.sort(key=lambda r: r.get("quality_score", 0.0), reverse=True)

    # Tool / label / score stats
    name_dist = Counter(r["observed_action"]["name"] for r in selected)
    label_dist = Counter(r.get("label") for r in selected)
    mean_q = (
        sum(r["quality_score"] for r in selected) / len(selected) if selected else 0.0
    )

    # Write JSONL
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {OUT_JSONL.relative_to(ROOT)}: {len(selected):,} records")

    # Splits
    splits = stratified_splits(selected)
    splits_payload = {
        "seed": SEED,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "stratification": "by (tool_name, label)",
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    with OUT_SPLITS.open("w", encoding="utf-8") as f:
        json.dump(splits_payload, f, ensure_ascii=False, indent=2)
    print(f"[write] {OUT_SPLITS.relative_to(ROOT)}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # Print summary
    print()
    print("=== summary ===")
    print(f"total: {len(selected):,}")
    print("tool distribution:")
    total_n = max(1, len(selected))
    for tool, n in name_dist.most_common():
        pct = 100.0 * n / total_n
        print(f"  {tool:14s} {n:5d}  ({pct:5.1f}%)")
    print("label distribution:")
    for lab, n in label_dist.most_common():
        pct = 100.0 * n / total_n
        print(f"  {str(lab):16s} {n:5d}  ({pct:5.1f}%)")
    print(f"mean quality_score: {mean_q:.4f}")
    print(
        f"splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
