"""Build a short, prepend-able persona prompt prefix from a profile dict.

The prefix is intentionally compact (<=500 chars) and DATA-DRIVEN: each
claim is composed from the profile histograms, never invented. Loading
more does not help small context windows on the LM and risks distracting
from the task prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parent / "profile.json"


def _top_names(items, k):
    return [n for n, _ in (items or [])[:k]]


def build_persona_prefix(profile: dict, max_chars: int = 500) -> str:
    bash_verbs = profile.get("bash_verb_top20") or []
    bash_top3 = _top_names(bash_verbs, 3)
    paths_top3 = _top_names(profile.get("path_histogram"), 3)
    czech_top3 = _top_names(
        profile.get("language_signals", {}).get("czech_token_counts"), 3
    )
    repos = _top_names(profile.get("repo_top5"), 1)
    recoveries = profile.get("recovery_top5") or []

    # Top distinctive recovery (skip identity Bash->Bash retries).
    distinctive = next(
        (r for r in recoveries if r.get("failed") != r.get("recovered")),
        None,
    )

    # Pick the dominant package manager from the actual histogram so the
    # prefix never invents a tool preference (e.g. claiming bun over pnpm
    # when bun never appears in the data).
    pm_candidates = ("pnpm", "npm", "yarn", "bun")
    pm_counts = {n: c for n, c in bash_verbs if n in pm_candidates}
    package_mgr = max(pm_counts, key=pm_counts.get) if pm_counts else None

    # Same for grep-vs-rg: report whichever the data actually shows on top.
    search_counts = {n: c for n, c in bash_verbs if n in ("grep", "rg")}
    search_tool = max(search_counts, key=search_counts.get) if search_counts else None

    lines = ["PERSONA NOTES (the user you are assisting):"]
    if bash_top3:
        lines.append(f"- Bash verbs (most-used): {', '.join(bash_top3)}.")
    if package_mgr or search_tool:
        bits = []
        if package_mgr:
            bits.append(f"package mgr: {package_mgr}")
        if search_tool:
            bits.append(f"search tool: {search_tool}")
        lines.append("- Observed defaults: " + ", ".join(bits) + ".")
    if paths_top3:
        lines.append(f"- Workspace paths: {', '.join(paths_top3)}.")
    if repos:
        lines.append(f"- Primary repo: {repos[0]}.")
    if czech_top3:
        lines.append(
            f"- User course-corrects in Czech ({', '.join(czech_top3)}) - treat as authoritative override."
        )
    if distinctive:
        lines.append(
            f"- On failed {distinctive['failed']}, pivots to {distinctive['recovered']} (inspect, don't retry blindly)."
        )

    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 3] + "..."
    return out


def inject_persona(system_prompt: str, profile: dict) -> str:
    prefix = build_persona_prefix(profile)
    if not system_prompt:
        return prefix
    return f"{prefix}\n\n{system_prompt}"


def load_profile(path: Path | str = PROFILE_PATH) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    p = load_profile()
    pre = build_persona_prefix(p)
    print(f"--- prefix ({len(pre)} chars) ---")
    print(pre)
