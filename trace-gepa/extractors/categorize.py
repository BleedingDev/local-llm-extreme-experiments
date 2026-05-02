from __future__ import annotations

import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CC_PATH = DATA / "cc_dataset.jsonl"
CODEX_PATH = DATA / "codex_dataset.jsonl"
OUT_PATH = DATA / "dataset.jsonl"
SPLITS_PATH = DATA / "splits.json"

HALLUC_MARKERS = (
    "doesn't exist", "does not exist", "no such", "not found",
    "not a real", "invented", "nesmysl", "did you mean",
)


def wait_for_inputs() -> tuple[bool, bool]:
    cc_ok = CC_PATH.exists()
    codex_ok = CODEX_PATH.exists()
    for _ in range(5):
        if cc_ok and codex_ok:
            break
        time.sleep(5)
        cc_ok = CC_PATH.exists()
        codex_ok = CODEX_PATH.exists()
    if not cc_ok and not codex_ok:
        print("[categorize] FATAL: neither cc_dataset.jsonl nor codex_dataset.jsonl present", file=sys.stderr)
        sys.exit(1)
    return cc_ok, codex_ok


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(orjson.loads(line))
            except orjson.JSONDecodeError:
                continue
    return out


def levenshtein(a: str, b: str, cap: int = 60) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        best = curr[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if curr[j] < best:
                best = curr[j]
        if best > cap:
            return cap + 1
        prev = curr
    return prev[lb]


def near_identical(a: str, b: str) -> bool:
    if not a or not b:
        return False
    try:
        return levenshtein(a, b, cap=30) < 30
    except Exception:
        return a[:80] == b[:80]


def session_id(rec: dict) -> str:
    sp = rec.get("src_path") or ""
    rid = rec.get("id") or ""
    if "_evt" in rid:
        return rid.rsplit("_evt", 1)[0]
    return sp


def get_input_str(rec: dict) -> str:
    obs = rec.get("observed_action") or {}
    inp = obs.get("input")
    if inp is None:
        return ""
    if isinstance(inp, str):
        return inp
    try:
        return orjson.dumps(inp).decode()
    except Exception:
        return str(inp)


def recategorize(records: list[dict]) -> None:
    by_session: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_session[session_id(r)].append(i)

    for r in records:
        if r.get("failure_category") is not None:
            continue
        label = r.get("label")
        obs = r.get("observed_action") or {}
        next_user = (r.get("next_user_message") or "").lower()
        kind = obs.get("kind")
        name = obs.get("name")
        is_err = bool(obs.get("result_is_error"))
        inp_str = get_input_str(r)

        if label == "user_corrected":
            r["failure_category"] = "user_correction"
            continue

        if next_user and any(m in next_user for m in HALLUC_MARKERS):
            r["failure_category"] = "hallucinated_path"
            continue

        if kind == "skill" and is_err:
            r["failure_category"] = "hallucinated_skill"
            continue

        if name in ("Agent", "Task") and len(inp_str) < 200:
            r["failure_category"] = "subagent_terse_prompt"
            continue

    for sid, idxs in by_session.items():
        idxs.sort(key=lambda i: records[i].get("src_event_idx", i))
        for k in range(1, len(idxs)):
            cur = records[idxs[k]]
            if cur.get("failure_category") is not None:
                continue
            prev = records[idxs[k - 1]]
            if prev.get("label") != "bad":
                continue
            if near_identical(get_input_str(prev), get_input_str(cur)):
                cur["failure_category"] = "retry_loop"


def stratified_split(records: list[dict], seed: int = 42) -> dict:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r.get("failure_category") or "_none_"].append(r)

    train, val, test = [], [], []
    for cat, items in by_cat.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    def dist(rs: list[dict], key: str) -> dict:
        c: Counter = Counter()
        for r in rs:
            c[r.get(key) or "null"] += 1
        return dict(c.most_common())

    return {
        "seed": seed,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "label_distribution": {
            "train": dist(train, "label"),
            "val": dist(val, "label"),
            "test": dist(test, "label"),
        },
        "category_distribution": {
            "train": dist(train, "failure_category"),
            "val": dist(val, "failure_category"),
            "test": dist(test, "failure_category"),
        },
        "ids": {
            "train": [r["id"] for r in train],
            "val": [r["id"] for r in val],
            "test": [r["id"] for r in test],
        },
    }


def main() -> None:
    wait_for_inputs()
    records: list[dict] = []
    records.extend(load_jsonl(CC_PATH))
    records.extend(load_jsonl(CODEX_PATH))
    if not records:
        print("[categorize] FATAL: input files exist but contained zero records", file=sys.stderr)
        sys.exit(1)

    pre_cat = Counter((r.get("failure_category") or "null") for r in records)
    recategorize(records)
    post_cat = Counter((r.get("failure_category") or "null") for r in records)
    label_dist = Counter((r.get("label") or "null") for r in records)

    DATA.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")

    splits = stratified_split(records, seed=42)
    with SPLITS_PATH.open("wb") as f:
        f.write(orjson.dumps(splits, option=orjson.OPT_INDENT_2))

    print(f"[categorize] total records: {len(records)}")
    print(f"[categorize] label distribution: {dict(label_dist.most_common())}")
    print(f"[categorize] category before: {dict(pre_cat.most_common(10))}")
    print(f"[categorize] category after:  {dict(post_cat.most_common(10))}")
    print(f"[categorize] split counts: {splits['counts']}")
    print(f"[categorize] wrote {OUT_PATH}")
    print(f"[categorize] wrote {SPLITS_PATH}")


if __name__ == "__main__":
    main()
