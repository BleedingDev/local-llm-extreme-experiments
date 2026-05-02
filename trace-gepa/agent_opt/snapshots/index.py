"""SQLite index for workflow snapshots."""
from __future__ import annotations

import sqlite3, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DEFAULT_INDEX_DIR = Path.home() / ".claude" / "snapshots"
DEFAULT_INDEX_PATH = DEFAULT_INDEX_DIR / "index.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id TEXT PRIMARY KEY, captured_at REAL, workspace TEXT, git_head TEXT,
    branch TEXT, prompt_hash TEXT, label TEXT, archive_path TEXT,
    size_bytes INTEGER, dirty_files INTEGER
)
"""
COLS = "id, captured_at, workspace, git_head, branch, prompt_hash, label, archive_path, size_bytes, dirty_files"


@dataclass
class SnapshotMeta:
    id: str
    captured_at: float
    workspace: str
    git_head: str
    branch: str
    prompt_hash: str
    label: str
    archive_path: str
    size_bytes: int
    dirty_files: int


def _connect(index_path: Path = DEFAULT_INDEX_PATH) -> sqlite3.Connection:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_path))
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def register(meta: SnapshotMeta, index_path: Path = DEFAULT_INDEX_PATH) -> None:
    conn = _connect(index_path)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO snapshots ({COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (meta.id, meta.captured_at, meta.workspace, meta.git_head, meta.branch,
             meta.prompt_hash, meta.label, meta.archive_path, meta.size_bytes, meta.dirty_files),
        )
        conn.commit()
    finally:
        conn.close()


def list_snapshots(limit: Optional[int] = None, label: Optional[str] = None,
                   index_path: Path = DEFAULT_INDEX_PATH) -> list[SnapshotMeta]:
    conn = _connect(index_path)
    try:
        sql = f"SELECT {COLS} FROM snapshots"
        args: list = []
        if label:
            sql += " WHERE label = ?"
            args.append(label)
        sql += " ORDER BY captured_at DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [SnapshotMeta(*r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def get(sid: str, index_path: Path = DEFAULT_INDEX_PATH) -> Optional[SnapshotMeta]:
    conn = _connect(index_path)
    try:
        row = conn.execute(f"SELECT {COLS} FROM snapshots WHERE id = ?", (sid,)).fetchone()
        return SnapshotMeta(*row) if row else None
    finally:
        conn.close()


def purge_older_than(days: float, index_path: Path = DEFAULT_INDEX_PATH) -> int:
    cutoff = time.time() - days * 86400.0
    conn = _connect(index_path)
    try:
        rows = conn.execute(
            "SELECT id, archive_path FROM snapshots WHERE captured_at < ?", (cutoff,)
        ).fetchall()
        for _id, ap in rows:
            try:
                Path(ap).unlink(missing_ok=True)
            except OSError:
                pass
        conn.execute("DELETE FROM snapshots WHERE captured_at < ?", (cutoff,))
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def meta_to_dict(meta: SnapshotMeta) -> dict:
    return asdict(meta)
