"""Tests for workflow snapshot capture."""
from __future__ import annotations

import gzip, sqlite3, subprocess, tarfile, time
from io import BytesIO
from pathlib import Path

import pytest

from . import capture as capture_mod
from .capture import capture
from .cli import main as cli_main
from .index import list_snapshots, purge_older_than

try:
    import zstandard as zstd  # type: ignore
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


def _read_archive_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix == ".zst":
        return zstd.ZstdDecompressor().decompress(raw, max_output_size=200_000_000)
    if path.suffix == ".gz":
        return gzip.decompress(raw)
    return raw


def _read_concat(path: Path) -> bytes:
    out = bytearray()
    with tarfile.open(fileobj=BytesIO(_read_archive_bytes(path)), mode="r") as tar:
        for m in tar.getmembers():
            f = tar.extractfile(m)
            if f is not None:
                out.extend(f.read())
                out.extend(b"\n")
    return bytes(out)


def _init_repo(tmp: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp, check=True)
    (tmp / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=tmp, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp, check=True)


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    snap_dir = tmp_path / "snaps"
    idx = snap_dir / "index.sqlite"
    monkeypatch.setattr(capture_mod, "DEFAULT_INDEX_DIR", snap_dir)
    monkeypatch.setattr(capture_mod, "DEFAULT_INDEX_PATH", idx)
    import agent_opt.snapshots.index as idx_mod
    monkeypatch.setattr(idx_mod, "DEFAULT_INDEX_PATH", idx)
    monkeypatch.setattr(idx_mod, "DEFAULT_INDEX_DIR", snap_dir)
    return snap_dir, idx


def test_capture_creates_archive(tmp_path, isolated_index):
    snap_dir, idx = isolated_index
    ws = tmp_path / "ws"; ws.mkdir(); _init_repo(ws)
    meta = capture(workspace_dir=ws, prompt="hello world", label="t1")
    assert Path(meta.archive_path).is_file()
    assert meta.size_bytes > 0
    rows = list_snapshots(index_path=idx)
    assert len(rows) == 1 and rows[0].id == meta.id


def test_capture_sanitises_env(tmp_path, isolated_index, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-DEADBEEFSECRETKEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRETXX")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SECRETZ")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "AWSSECRETT")
    monkeypatch.setenv("BAG_TEST_VAR", "ok-keep-me")
    ws = tmp_path / "ws"; ws.mkdir(); _init_repo(ws)
    meta = capture(workspace_dir=ws, prompt="x", label="env")
    blob = _read_concat(Path(meta.archive_path))
    for s in (b"sk-DEADBEEFSECRETKEY", b"sk-ant-SECRETXX", b"ghp_SECRETZ", b"AWSSECRETT"):
        assert s not in blob
    assert b"ok-keep-me" in blob


def test_capture_dirty_diff(tmp_path, isolated_index):
    ws = tmp_path / "ws"; ws.mkdir(); _init_repo(ws)
    (ws / "README.md").write_text("hello\nNEW_DIRTY_LINE\n")
    meta = capture(workspace_dir=ws, prompt=None, label="dirty")
    assert b"NEW_DIRTY_LINE" in _read_concat(Path(meta.archive_path))
    assert meta.dirty_files >= 1


def test_list_after_capture(tmp_path, isolated_index):
    snap_dir, idx = isolated_index
    ws = tmp_path / "ws"; ws.mkdir(); _init_repo(ws)
    capture(workspace_dir=ws, prompt="a", label="L")
    capture(workspace_dir=ws, prompt="b", label="L")
    assert len(list_snapshots(index_path=idx)) == 2


def test_purge_older_than(tmp_path, isolated_index):
    snap_dir, idx = isolated_index
    ws = tmp_path / "ws"; ws.mkdir(); _init_repo(ws)
    m1 = capture(workspace_dir=ws, prompt="x", label="old")
    m2 = capture(workspace_dir=ws, prompt="y", label="new")
    conn = sqlite3.connect(str(idx))
    conn.execute("UPDATE snapshots SET captured_at = ? WHERE id = ?",
                 (time.time() - 100 * 86400, m1.id))
    conn.commit(); conn.close()
    assert purge_older_than(days=30, index_path=idx) == 1
    remaining = list_snapshots(index_path=idx)
    assert len(remaining) == 1 and remaining[0].id == m2.id
    assert not Path(m1.archive_path).exists()


def test_replay_not_implemented(tmp_path, isolated_index, capsys):
    rc = cli_main(["replay", "deadbeef"])
    assert rc == 2
    assert "TBD" in capsys.readouterr().out
