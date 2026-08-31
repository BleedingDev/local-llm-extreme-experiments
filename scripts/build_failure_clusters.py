#!/usr/bin/env python3
"""Cluster verifier failures from BAG trials by signature similarity.

Walk all losing trials in `bench/.bag/optimizer/dataset.jsonl` plus any
fresh trials under `bench/jobs/` (whose verifier/test-stdout.txt is
present), normalize the verifier output, extract a "failure signature",
cluster signatures (exact-match + character-trigram Jaccard), and emit
`bench/.bag/optimizer/failure-clusters.json`.

Pure stdlib. No embedding deps.

Usage:
    python3 scripts/build_failure_clusters.py \
        [--repo-root .] \
        [--output bench/.bag/optimizer/failure-clusters.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VERIFIER_TAIL_CHARS = 4000
SIGNATURE_MAX_LEN = 320
EXEMPLAR_MAX_LEN = 800
JACCARD_THRESHOLD = 0.6
TRIGRAM_N = 3

# --- normalization regexes ---------------------------------------------------
RX_TIMESTAMP_ISO = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
RX_TIMESTAMP_HMS = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[,.]\d+)?\b")
RX_HEX_HASH = re.compile(r"\b[0-9a-fA-F]{8,40}\b")
RX_TMP_PATH = re.compile(r"/tmp/[A-Za-z0-9._-]+")
RX_VAR_FOLDERS = re.compile(r"/var/folders/[A-Za-z0-9_/+.-]+")
RX_PYBYTECODE = re.compile(r"__pycache__/[^\s'\"]+\.pyc")
RX_CONTAINER_ID = re.compile(r"\b[a-f0-9]{12}\b")
RX_LINENO = re.compile(r":(\d+)(?=[:\s])")
RX_PORT_RANDOM = re.compile(r":(\b[3-6]\d{4}\b)")
RX_USER_PATH = re.compile(r"/Users/[^\s/'\"]+")
RX_HOME_PATH = re.compile(r"/home/[^\s/'\"]+")
RX_SESSION_ID = re.compile(r"\b[a-zA-Z0-9_-]{20,}\b(?=[\s'\"\)])")
RX_MULTISPACE = re.compile(r"\s+")

# Salient identifier extraction
RX_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
RX_HTTP_CODE = re.compile(r"\bHTTP\s*(\d{3})\b", re.IGNORECASE)
RX_EXIT_CODE = re.compile(r"\b(?:exit_code|exit code|returncode)\D*(\d+)", re.IGNORECASE)

GENERIC_TOKENS = {
    "self", "test", "tests", "True", "False", "None", "assert", "Error",
    "Exception", "the", "and", "for", "not", "with", "from", "import",
    "is", "in", "of", "to", "be", "no", "an", "at", "on", "by", "as",
    "raise", "raised", "raises", "should", "expected", "got", "found",
    "value", "values", "result", "results", "output", "outputs",
    "got", "but", "if", "while", "this", "that", "those", "these",
    "AssertionError", "FileNotFoundError", "ModuleNotFoundError",
}


# --- helpers -----------------------------------------------------------------
def warn(msg: str) -> None:
    print(f"warn: {msg}", file=sys.stderr)


def read_text_tail(path: Path, n_chars: int) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warn(f"failed to read {path}: {exc}")
        return None
    return text if len(text) <= n_chars else text[-n_chars:]


def normalize(text: str) -> str:
    """Strip volatile bits so equivalent failures collapse together."""
    t = text
    t = RX_TIMESTAMP_ISO.sub("<TS>", t)
    t = RX_TIMESTAMP_HMS.sub("<TS>", t)
    t = RX_USER_PATH.sub("<USER>", t)
    t = RX_HOME_PATH.sub("<HOME>", t)
    t = RX_TMP_PATH.sub("<TMP>", t)
    t = RX_VAR_FOLDERS.sub("<VARFOLDER>", t)
    t = RX_PYBYTECODE.sub("<PYC>", t)
    t = RX_HEX_HASH.sub("<HASH>", t)
    t = RX_CONTAINER_ID.sub("<CID>", t)
    t = RX_LINENO.sub(":<L>", t)
    t = RX_PORT_RANDOM.sub(":<PORT>", t)
    return t


def extract_signature(stdout_tail: str) -> str:
    """Pick the most informative single line(s) from the verifier tail.

    Priority (most → least specific):
      1. pytest 'E   <Exception>: <detail>' lines (these contain the real
         error message with arguments).
      2. raw 'AssertionError: ...' / 'FooError: ...' lines outside the
         pytest 'E   ' marker.
      3. pytest summary 'FAILED ...' line.
      4. last 3 non-empty lines joined.
    """
    lines = [ln.rstrip() for ln in stdout_tail.splitlines()]
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return ""

    # 1. pytest 'E   <something>' marker — prefer the LAST line that looks
    #    like an exception header (FooError: ...) or a top-level assert,
    #    not a hint continuation like 'Use -v to get more diff'.
    e_lines: list[str] = []
    for ln in nonempty:
        s = ln.strip()
        if s.startswith("E   ") or s.startswith("E\t"):
            sig = s[1:].strip()
            if sig and len(sig) > 5:
                e_lines.append(sig)
    if e_lines:
        # Walk from the end; pick the last "exception header" line.
        header_rx = re.compile(
            r"^(?:[A-Z][A-Za-z]*Error|[A-Z][A-Za-z]*Exception|assert\b)"
        )
        for sig in reversed(e_lines):
            if header_rx.search(sig):
                return sig[:SIGNATURE_MAX_LEN]
        # Fallback: last E-line that isn't a "where ..." / "assert ..." cont.
        for sig in reversed(e_lines):
            low = sig.lower()
            if low.startswith("+ ") or low.startswith("where ") or low.startswith("use -"):
                continue
            return sig[:SIGNATURE_MAX_LEN]
        return e_lines[-1][:SIGNATURE_MAX_LEN]

    # 2. raw '*Error:' line that's not just a re-print of FAILED summary
    for ln in reversed(nonempty):
        s = ln.strip()
        if s.lstrip().startswith("FAILED "):
            continue  # handled later
        if re.search(r"\b[A-Z][A-Za-z]*Error\b\s*:", s):
            return s[:SIGNATURE_MAX_LEN]

    # 3. last pytest "FAILED" line
    for ln in reversed(nonempty):
        if ln.lstrip().startswith("FAILED "):
            return ln.strip()[:SIGNATURE_MAX_LEN]

    # 4. last 3 non-empty lines joined
    tail = " | ".join(nonempty[-3:])
    return tail[:SIGNATURE_MAX_LEN]


def trigrams(text: str) -> set[str]:
    s = re.sub(r"\s+", " ", text.lower()).strip()
    if len(s) < TRIGRAM_N:
        return {s} if s else set()
    return {s[i : i + TRIGRAM_N] for i in range(len(s) - TRIGRAM_N + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def cluster_id_from_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60] if slug else "cluster"


def derive_name(signature: str, task_names: list[str]) -> str:
    """Pick a short identifier-ish phrase that summarizes the failure."""
    sig = signature
    # Boost: HTTP code, exit code
    m_http = RX_HTTP_CODE.search(sig)
    m_exit = RX_EXIT_CODE.search(sig)

    # Identifier tokens (camelCase, snake_case, longer than 3)
    idents = [
        tok for tok in RX_IDENT.findall(sig)
        if tok not in GENERIC_TOKENS and not tok.isdigit() and len(tok) >= 4
    ]
    # Prefer non-stopword identifiers seen first
    seen: list[str] = []
    seen_set: set[str] = set()
    for tok in idents:
        if tok.lower() in seen_set:
            continue
        seen_set.add(tok.lower())
        seen.append(tok)
        if len(seen) >= 3:
            break

    parts: list[str] = []
    if len(task_names) >= 3:
        parts.append("multi-task")
    elif task_names:
        parts.append(task_names[0])
    else:
        parts.append("task")

    if m_http:
        parts.append(f"http-{m_http.group(1)}")
    if m_exit:
        parts.append(f"exit-{m_exit.group(1)}")

    parts.extend(tok.lower() for tok in seen)

    if len(parts) <= 1:
        # fallback: short suffix of the signature
        suffix = re.sub(r"[^A-Za-z0-9]+", "-", sig.strip()[:30].lower()).strip("-")
        if suffix:
            parts.append(suffix)

    return cluster_id_from_name("-".join(parts))


# --- record loading ----------------------------------------------------------
def iter_dataset_records(dataset_path: Path) -> Iterable[dict[str, Any]]:
    if not dataset_path.is_file():
        return
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                warn(f"bad jsonl line in {dataset_path}: {exc}")


def is_loss(rec: dict[str, Any]) -> bool:
    if rec.get("exception_type"):
        return True
    reward = rec.get("reward")
    return reward is None or float(reward or 0.0) <= 0.0


def trial_paths(rec: dict[str, Any], repo_root: Path) -> tuple[str, Path | None]:
    trial_id = rec.get("trial_id") or "unknown"
    src = rec.get("source_paths") or {}
    result_path = src.get("result")
    if not result_path:
        return trial_id, None
    p = Path(result_path)
    if not p.is_absolute():
        p = repo_root / p
    stdout_path = p.parent / "verifier" / "test-stdout.txt"
    return trial_id, stdout_path if stdout_path.is_file() else None


def discover_jobs_trials(
    repo_root: Path,
    seen_trial_ids: set[str],
) -> Iterable[dict[str, Any]]:
    """Pick up trials under bench/jobs/ that are not already in dataset."""
    jobs_root = repo_root / "bench" / "jobs"
    if not jobs_root.is_dir():
        return
    for job_dir in sorted(jobs_root.iterdir()):
        if not job_dir.is_dir():
            continue
        for trial_dir in sorted(job_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            trial_id = trial_dir.name
            if trial_id in seen_trial_ids:
                continue
            verifier_path = trial_dir / "verifier" / "test-stdout.txt"
            if not verifier_path.is_file():
                continue
            reward_path = trial_dir / "verifier" / "reward.txt"
            try:
                reward_raw = reward_path.read_text(encoding="utf-8").strip() if reward_path.is_file() else ""
                reward = float(reward_raw) if reward_raw else 0.0
            except (OSError, ValueError):
                reward = 0.0
            if reward > 0.0:
                continue
            # extract task_name from trial_id prefix (task-name__suffix)
            task_name = trial_id.rsplit("__", 1)[0] if "__" in trial_id else trial_id
            mtime_iso = datetime.fromtimestamp(verifier_path.stat().st_mtime, tz=timezone.utc).isoformat()
            yield {
                "trial_id": trial_id,
                "task_name": task_name,
                "job_id": job_dir.name,
                "reward": reward,
                "exception_type": None,
                "_verifier_path_override": verifier_path,
                "_synthetic_iso": mtime_iso,
            }


# --- clustering --------------------------------------------------------------
def build_clusters(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Two-pass: exact-signature clusters, then trigram-Jaccard fuzzy merge."""
    # Pass 1: exact-match groups by normalized signature
    by_sig: dict[str, list[int]] = defaultdict(list)
    for idx, f in enumerate(failures):
        by_sig[f["signature"]].append(idx)

    seed_groups: list[list[int]] = list(by_sig.values())

    # Pass 2: merge groups whose lead signatures are similar enough
    grams_per_group: list[set[str]] = []
    for grp in seed_groups:
        sig = failures[grp[0]]["signature"]
        grams_per_group.append(trigrams(sig))

    merged: list[list[int]] = []
    used = [False] * len(seed_groups)
    for i, grp in enumerate(seed_groups):
        if used[i]:
            continue
        cur = list(grp)
        cur_gr = set(grams_per_group[i])
        used[i] = True
        # greedily absorb similar groups
        for j in range(i + 1, len(seed_groups)):
            if used[j]:
                continue
            if jaccard(cur_gr, grams_per_group[j]) >= JACCARD_THRESHOLD:
                cur.extend(seed_groups[j])
                cur_gr |= grams_per_group[j]
                used[j] = True
        merged.append(cur)

    # Sort: largest first
    merged.sort(key=lambda g: (-len(g), failures[g[0]]["signature"]))

    clusters: list[dict[str, Any]] = []
    name_seen: Counter[str] = Counter()
    for grp in merged:
        members = [failures[i] for i in grp]
        # Pick exemplar = longest signature member (most info)
        exemplar = max(members, key=lambda m: len(m["signature"]))
        signature = exemplar["signature"]
        tasks = sorted({m["task_name"] for m in members})
        first_seen = min((m.get("seen_at") or "") for m in members)
        last_seen = max((m.get("seen_at") or "") for m in members)
        base_name = derive_name(signature, tasks)
        # de-dup names
        name_seen[base_name] += 1
        name = base_name if name_seen[base_name] == 1 else f"{base_name}-{name_seen[base_name]}"
        clusters.append({
            "id": name,
            "name": name,
            "size": len(members),
            "trial_ids": sorted(m["trial_id"] for m in members),
            "signature": signature,
            "tasks": tasks,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "exemplar_verifier_excerpt": exemplar["excerpt"],
        })
    return clusters


# --- main --------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repo root (default: cwd).")
    parser.add_argument(
        "--dataset",
        default="bench/.bag/optimizer/dataset.jsonl",
        help="BAG optimizer dataset jsonl",
    )
    parser.add_argument(
        "--output",
        default="bench/.bag/optimizer/failure-clusters.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-cluster summary",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_path = repo_root / args.dataset
    output_path = repo_root / args.output

    failures: list[dict[str, Any]] = []
    seen_trial_ids: set[str] = set()

    for rec in iter_dataset_records(dataset_path):
        seen_trial_ids.add(rec.get("trial_id", ""))
        if not is_loss(rec):
            continue
        trial_id, stdout_path = trial_paths(rec, repo_root)
        if not stdout_path:
            # Fall back to stdout_tail in dataset
            tail = (rec.get("verifier") or {}).get("stdout_tail") or ""
            if not tail:
                continue
            tail_text = tail
        else:
            tail_text = read_text_tail(stdout_path, VERIFIER_TAIL_CHARS) or ""
            if not tail_text:
                continue
        normalized = normalize(tail_text)
        signature = extract_signature(normalized)
        if not signature:
            continue
        seen_at = rec.get("job_id") or ""
        excerpt = normalized[-EXEMPLAR_MAX_LEN:]
        failures.append({
            "trial_id": trial_id,
            "task_name": rec.get("task_name") or "unknown",
            "signature": signature,
            "excerpt": excerpt,
            "seen_at": seen_at,
        })

    # Pick up new trials not in dataset
    for rec in discover_jobs_trials(repo_root, seen_trial_ids):
        verifier_path = rec.get("_verifier_path_override")
        tail_text = read_text_tail(verifier_path, VERIFIER_TAIL_CHARS) or ""
        if not tail_text:
            continue
        normalized = normalize(tail_text)
        signature = extract_signature(normalized)
        if not signature:
            continue
        failures.append({
            "trial_id": rec["trial_id"],
            "task_name": rec["task_name"],
            "signature": signature,
            "excerpt": normalized[-EXEMPLAR_MAX_LEN:],
            "seen_at": rec.get("job_id") or rec.get("_synthetic_iso") or "",
        })

    clusters = build_clusters(failures)

    doc = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_failures": len(failures),
        "clusters": clusters,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")

    if not args.quiet:
        print(f"failures clustered: {len(failures)}", file=sys.stderr)
        print(f"distinct clusters:  {len(clusters)}", file=sys.stderr)
        print(f"output:             {output_path}", file=sys.stderr)
        for c in clusters[:10]:
            print(
                f"  size={c['size']:>3}  {c['name']}  ::  "
                f"{c['signature'][:80]}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
