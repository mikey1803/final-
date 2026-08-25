"""Repo-wide structural guard: no test may write outside its own tmp_path.

Every incident found this session (test_harness_smoke silently overwriting
results/phase1/, the six --smoke scripts silently overwriting results/tables/)
was found by manual enumeration -- reading code, checking mtimes, running
scripts by hand. Enumeration only catches what someone thought to look for.
This fixture is the structural backstop: it patches the primitives file
writes route through (builtins.open and io.open -- pathlib's
Path.write_text/write_bytes/open call io.open directly, not builtins.open;
pandas' to_csv/to_parquet were verified empirically to route through
builtins.open even for the pyarrow parquet engine -- plus sqlite3.connect,
a separate C-extension path that bypasses both open() and io.open() and was
caught missing this guard entirely on first use: test_memory_basic and
test_memory_selector_roundtrip both connect to real backend/data/*.db paths
and only self-clean via unlink() afterward, invisible to an open()-only
guard) and fails any write-capable call whose resolved path is inside the
repo but outside tmp_path.

KNOWN GAP: this only guards the pytest process itself. Tests that shell out
via subprocess.run (e.g. test_stationary_experiments.py's --smoke subprocess
calls) run in a separate Python process this monkeypatch cannot reach -- a
child process writing to a production path would NOT be caught here. That
class of bug needs the corresponding script's own default-path behavior
fixed (as done today for all six --smoke scripts), not this fixture. Any
other C-extension file I/O that bypasses both open()/io.open() and
sqlite3.connect() (e.g. a library using raw os.open or its own native file
handle) would have the same blind spot -- sqlite3 is patched because it's
the one instance actually found in this codebase, not because the list is
now exhaustive.
"""

from __future__ import annotations

import builtins
import io
import sqlite3
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # failure-aware-agent/
_SYSTEM_TMP = Path(tempfile.gettempdir()).resolve()

# Path components that make a write infrastructure noise, not a test writing
# real data -- never something this guard is meant to catch.
_ALWAYS_ALLOWED_COMPONENTS = {"__pycache__", ".pytest_cache", ".venv", "site-packages", ".git"}


def _is_write_mode(mode) -> bool:
    if not isinstance(mode, str):
        return False
    return any(c in mode for c in "wax+")


def _resolve_target(file) -> Path | None:
    if isinstance(file, int):
        return None  # already-open file descriptor, not a new path
    try:
        return Path(file).resolve()
    except TypeError:
        return None


def _is_allowed(path: Path, tmp_root: Path) -> bool:
    if not _within(path, _REPO_ROOT):
        return True  # outside the repo entirely: system temp, user caches, etc.
    if _within(path, tmp_root) or _within(path, _SYSTEM_TMP):
        return True
    if any(part in _ALWAYS_ALLOWED_COMPONENTS for part in path.parts):
        return True
    return False


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sqlite_target(database) -> Path | None:
    """Resolve a sqlite3.connect() `database` argument to a Path, or None
    if it names something with no real on-disk write target (:memory:, an
    explicit read-only URI)."""
    if not isinstance(database, str):
        return None
    if database == ":memory:" or database.startswith("file::memory:"):
        return None
    if database.startswith("file:") and "mode=ro" in database:
        return None  # explicit read-only URI, not a write target
    path_part = database[5:].split("?")[0] if database.startswith("file:") else database
    try:
        return Path(path_part).resolve()
    except (TypeError, ValueError):
        return None


@pytest.fixture(autouse=True)
def _forbid_writes_outside_tmp_path(tmp_path, monkeypatch):
    tmp_root = tmp_path.resolve()
    real_open = builtins.open
    real_sqlite_connect = sqlite3.connect

    def guarded_open(file, mode="r", *args, **kwargs):
        if _is_write_mode(mode):
            target = _resolve_target(file)
            if target is not None and not _is_allowed(target, tmp_root):
                raise AssertionError(
                    f"test attempted to write outside tmp_path: {target}\n"
                    f"(mode={mode!r}) -- use tmp_path (or an explicit output/results_dir "
                    f"override) instead of a real repo path."
                )
        return real_open(file, mode, *args, **kwargs)

    def guarded_connect(database, *args, **kwargs):
        target = _sqlite_target(database)
        if target is not None and not _is_allowed(target, tmp_root):
            raise AssertionError(
                f"test attempted to open a sqlite3 database outside tmp_path: {target}\n"
                f"-- use tmp_path for test databases instead of a real repo path."
            )
        return real_sqlite_connect(database, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
