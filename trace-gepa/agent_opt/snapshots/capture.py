"""Capture a reproducible snapshot of a workspace's start-state."""
from __future__ import annotations

import hashlib, io, json, os, re, subprocess, tarfile, time, uuid
from pathlib import Path
from typing import Optional

from .index import DEFAULT_INDEX_DIR, DEFAULT_INDEX_PATH, SnapshotMeta, register

try:
    import zstandard as zstd  # type: ignore
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    import gzip

DROP_ENV_EXACT = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "HF_TOKEN"}
DROP_ENV_PREFIXES = ("AWS_",)
WHITELIST_PREFIXES = ("BAG_", "CLAUDE_")
WHITELIST_EXACT = {"PATH", "SHELL", "LANG", "PWD", "ANTHROPIC_AUTH_TOKEN"}
REDACT = {"ANTHROPIC_AUTH_TOKEN"}
PATH_TRUNCATE = 512
PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\-]+/[A-Za-z0-9_./\-]+")


def _run(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_state(ws: Path) -> dict:
    head = _run(["git", "rev-parse", "HEAD"], ws).strip()
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], ws).strip()
    diff = _run(["git", "diff", "HEAD"], ws)
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], ws)
    n = sum(1 for L in diff.splitlines() if L.startswith("diff --git"))
    return {"head": head, "branch": branch, "diff": diff, "untracked": untracked, "dirty_files": n}


def _scan_paths(prompt: str, ws: Path) -> list[Path]:
    if not prompt:
        return []
    found, seen = [], set()
    for tok in PATH_TOKEN_RE.findall(prompt):
        if tok in seen:
            continue
        seen.add(tok)
        c = (ws / tok).resolve() if not Path(tok).is_absolute() else Path(tok)
        if c.is_file():
            found.append(c)
    return found[:50]


def _file_meta(p: Path) -> dict:
    try:
        data = p.read_bytes()
    except OSError as e:
        return {"path": str(p), "error": str(e)}
    return {"path": str(p), "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data), "head_b64_first_1000": data[:1000].hex()}


def _filter_env() -> dict:
    out = {}
    for k, v in os.environ.items():
        if k in DROP_ENV_EXACT or any(k.startswith(p) for p in DROP_ENV_PREFIXES):
            continue
        if not (any(k.startswith(p) for p in WHITELIST_PREFIXES) or k in WHITELIST_EXACT):
            continue
        if k in REDACT:
            out[k] = "<REDACTED>"
        elif k == "PATH":
            out[k] = v[:PATH_TRUNCATE]
        else:
            out[k] = v
    return out


def _claude_excerpt() -> dict:
    cj = Path.home() / ".claude.json"
    if not cj.is_file():
        return {}
    try:
        data = json.loads(cj.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict = {"top_level_keys": sorted(data.keys())}
    if isinstance(data.get("model"), str):
        out["model"] = data["model"]
    mcp = data.get("mcpServers")
    if isinstance(mcp, dict):
        out["mcp_server_keys"] = sorted(mcp.keys())
    return out


def _package_metadata(ws: Path) -> dict:
    out: dict = {}
    for name in ("package.json", "pyproject.toml", "requirements.txt"):
        p = ws / name
        if not p.is_file():
            continue
        try:
            content = p.read_text(errors="replace")
        except OSError:
            continue
        if name == "package.json":
            try:
                obj = json.loads(content)
                out[name] = {k: obj.get(k) for k in ("name", "version", "dependencies", "devDependencies", "scripts")}
                continue
            except json.JSONDecodeError:
                pass
        out[name] = content[:8000]
    return out


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


def capture(workspace_dir: Path, prompt: Optional[str] = None, label: Optional[str] = None,
            max_total_bytes: int = 100_000_000, snapshot_dir: Optional[Path] = None,
            index_path: Optional[Path] = None) -> SnapshotMeta:
    """Capture workspace snapshot. Returns SnapshotMeta; writes archive + sqlite row."""
    from . import capture as _self  # late lookup so monkeypatch can override module attrs
    snapshot_dir = snapshot_dir or _self.DEFAULT_INDEX_DIR
    index_path = index_path or _self.DEFAULT_INDEX_PATH
    ws = Path(workspace_dir).resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    sid = uuid.uuid4().hex[:12]
    captured_at = time.time()
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(captured_at))
    ext = "tar.zst" if HAS_ZSTD else "tar.gz"
    archive_path = snapshot_dir / f"{sid}-{ts}.{ext}"

    git = _git_state(ws)
    files_meta = [_file_meta(p) for p in _scan_paths(prompt or "", ws)]
    env = _filter_env()
    claude = _claude_excerpt()
    pkg = _package_metadata(ws)
    prompt_hash = hashlib.sha256((prompt or "").encode()).hexdigest()

    meta_json = {
        "id": sid, "captured_at": captured_at, "workspace": str(ws),
        "label": label or "", "prompt_hash": prompt_hash,
        "claude_code_version": os.environ.get("CLAUDE_CODE_VERSION", ""),
        "os": os.uname().sysname if hasattr(os, "uname") else "",
        "compression": "zstd" if HAS_ZSTD else "gzip",
    }

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        _add(tar, "meta.json", json.dumps(meta_json, indent=2).encode())
        _add(tar, "prompt.txt", (prompt or "").encode())
        _add(tar, "git.head", f"{git['head']}\n{git['branch']}\n".encode())
        _add(tar, "git.diff", git["diff"].encode())
        _add(tar, "git.untracked", git["untracked"].encode())
        _add(tar, "files.json", json.dumps(files_meta, indent=2).encode())
        _add(tar, "env.json", json.dumps(env, indent=2).encode())
        _add(tar, "claude.excerpt.json", json.dumps(claude, indent=2).encode())
        _add(tar, "packages.json", json.dumps(pkg, indent=2).encode())
        if raw.tell() > max_total_bytes:
            raise RuntimeError(f"snapshot exceeds {max_total_bytes} bytes")

    raw_bytes = raw.getvalue()
    if HAS_ZSTD:
        archive_path.write_bytes(zstd.ZstdCompressor(level=19).compress(raw_bytes))
    else:
        with gzip.open(archive_path, "wb", compresslevel=9) as f:
            f.write(raw_bytes)

    meta = SnapshotMeta(id=sid, captured_at=captured_at, workspace=str(ws),
                        git_head=git["head"], branch=git["branch"], prompt_hash=prompt_hash,
                        label=label or "", archive_path=str(archive_path),
                        size_bytes=archive_path.stat().st_size, dirty_files=git["dirty_files"])
    register(meta, index_path=index_path)
    return meta
