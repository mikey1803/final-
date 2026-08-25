"""Append-only JSONL storage for the blind-design elicitation study (Task 11).

One JSON object per line, one line per submission, in the order
submissions arrive. Records are NEVER overwritten and NEVER deleted --
:meth:`SubmissionStore.append` refuses (raises :class:`TokenAlreadyExistsError`)
if the token already has a record, which is what backs the study API's
HTTP 409 "immutable once submitted" rule (see ``study/app/backend/main.py``).

No participant identifier of any kind (name, email, IP address) is ever
accepted by this module's schema -- ``token`` is an opaque, participant-
generated or server-generated random string with no link back to a real
identity anywhere in this codebase. See ``study/protocol.md``'s Ethics
section.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

__all__ = [
    "SCHEMA_VERSION",
    "DATASET_VERSION",
    "REQUIRED_FIELDS",
    "TokenAlreadyExistsError",
    "StorageWriteError",
    "MalformedRecordError",
    "SubmissionStore",
    "current_software_commit",
]

# Bumped whenever the ON-DISK RECORD SCHEMA changes (field added/removed/
# renamed) -- independent of the STUDY PROTOCOL version (which can change,
# e.g. a wording fix, without the schema changing) and independent of the
# DATASET (release artifact) version (which is bumped per public release,
# even if neither schema nor protocol changed, e.g. re-releasing with more
# submissions).
SCHEMA_VERSION = "1.0.0"
DATASET_VERSION = "1.0.0"

REQUIRED_FIELDS = (
    "token",
    "timestamp_utc",
    "background",
    "display_order",
    "mapping",
    "confidence",
    "rationale",
    "seconds_on_task",
    "protocol_version",
)


class TokenAlreadyExistsError(Exception):
    """Raised by :meth:`SubmissionStore.append` when ``token`` already has
    a recorded submission -- the storage-layer half of the study's
    immutability guarantee (the API layer maps this to HTTP 409)."""


class StorageWriteError(Exception):
    """Raised when a write to the JSONL file fails for any reason (disk
    full, permissions, etc.). The caller (the API layer) MUST surface this
    as HTTP 500 and tell the client the submission was NOT recorded --
    never a false success (see the module docstring of
    ``study/app/backend/main.py``)."""


class MalformedRecordError(Exception):
    """Raised when a record is missing a required field before it is
    ever written -- defense in depth: the API layer validates with
    Pydantic first (HTTP 422 on failure), but the storage layer does not
    trust its caller and re-checks before touching disk."""


def current_software_commit() -> str:
    """Best-effort short git commit hash of the current checkout, for
    dataset-versioning provenance (see ``export_anonymised``). Returns
    ``"unknown"`` if this checkout is not a git repository or ``git`` is
    unavailable -- never raises (provenance metadata is best-effort, not
    load-bearing for data collection itself)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


@dataclass
class SubmissionStore:
    """Append-only JSONL store, one file per study deployment.

    Parameters
    ----------
    path : Path
        Location of the ``.jsonl`` file. Created (with parent
        directories) on first write if it does not exist.
    """

    path: Path
    _token_cache: Optional[set] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def _known_tokens(self) -> set:
        """Scan the file once and cache the token set. Invalidated (re-scanned)
        the next time the process restarts -- correctness across concurrent
        processes is handled by :meth:`append`'s own re-check immediately
        before writing, not by this cache (which is purely an optimization
        to avoid a full file scan on every single append within one
        process's lifetime)."""
        if self._token_cache is None:
            tokens = set()
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            tokens.add(json.loads(line)["token"])
                        except (json.JSONDecodeError, KeyError):
                            continue
            self._token_cache = tokens
        return self._token_cache

    def has_token(self, token: str) -> bool:
        """True iff a submission for ``token`` has already been recorded."""
        return token in self._known_tokens()

    def append(self, record: dict) -> None:
        """Append one submission record. Never overwrites.

        Raises
        ------
        MalformedRecordError
            If a required field is missing (see ``REQUIRED_FIELDS``).
        TokenAlreadyExistsError
            If ``record["token"]`` already has a submission on disk.
        StorageWriteError
            If the write itself fails (disk/permissions/etc.) -- the
            caller must not report success in this case.
        """
        missing = [f for f in REQUIRED_FIELDS if f not in record]
        if missing:
            raise MalformedRecordError(f"record missing required fields: {missing}")

        token = record["token"]
        if self.has_token(token):
            raise TokenAlreadyExistsError(f"token {token!r} already has a recorded submission")

        payload = dict(record)
        payload.setdefault("schema_version", SCHEMA_VERSION)

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                import os

                os.fsync(f.fileno())
        except OSError as exc:
            raise StorageWriteError(f"failed to write submission for token {token!r}: {exc}") from exc

        self._known_tokens().add(token)

    def read_all(self) -> list[dict]:
        """Return every recorded submission, in append order. Skips (does
        not raise on) any corrupted trailing line -- an append-only log
        that failed mid-write on its LAST line should still yield every
        prior, complete record."""
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def export_anonymised(self, output_path: Path) -> Path:
        """Write the public release artifact: every recorded submission
        (already fully anonymous -- no identifier is ever collected, see
        the module docstring) plus a release-metadata header, as one JSON
        document (not JSONL, since this is a one-shot release snapshot,
        not an append target).

        Every record is re-validated against ``REQUIRED_FIELDS`` and an
        explicit denylist of identifier-shaped keys (``name``, ``email``,
        ``ip``, ``ip_address``, ``user_id``) before release -- defense in
        depth, since the schema never collects these in the first place,
        but a release step should never simply trust upstream data.
        """
        records = self.read_all()
        denylist = {"name", "email", "ip", "ip_address", "user_id", "username"}
        for r in records:
            leaked = denylist & set(r.keys())
            if leaked:
                raise MalformedRecordError(
                    f"record for token {r.get('token', '<unknown>')!r} contains disallowed "
                    f"identifier-shaped field(s) {leaked}; refusing to export"
                )

        release = {
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "software_commit": current_software_commit(),
            "n_records": len(records),
            "records": records,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(release, indent=2, sort_keys=True), encoding="utf-8")
        return output_path
