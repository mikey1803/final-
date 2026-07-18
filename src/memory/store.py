"""SQLite-backed persistent outcome memory store.

Records every recovery-strategy outcome to a local SQLite database so
that the AdaptiveSelector can resume learning across runs instead of
starting cold each time.  Also provides aggregation helpers for the
dashboard and evaluation harness.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


_DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "data" / "outcomes.db"


class OutcomeMemory:
    """Persistent store for recovery-strategy outcome records.

    Args:
        db_path: Path to the SQLite database file.  Parent directories
            are created automatically if they don't exist.  Defaults to
            ``data/outcomes.db`` relative to the project root.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_outcomes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                failure_type    TEXT    NOT NULL,
                strategy        TEXT    NOT NULL,
                success         INTEGER NOT NULL,
                cost            REAL    NOT NULL DEFAULT 1.0,
                episode_number  INTEGER NOT NULL,
                timestamp       TEXT    NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ft ON strategy_outcomes(failure_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_st ON strategy_outcomes(strategy)
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        failure_type: str,
        strategy: str,
        success: bool,
        cost: float = 1.0,
        episode_number: int = 0,
    ) -> None:
        """Append a single outcome row to the database.

        Args:
            failure_type: The ``FailureType.value`` string.
            strategy: The ``RecoveryStrategy.value`` string.
            success: Whether the recovery succeeded.
            cost: Cost weight of this attempt.
            episode_number: The episode in which this occurred.
        """
        self._conn.execute(
            "INSERT INTO strategy_outcomes "
            "(failure_type, strategy, success, cost, episode_number, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                failure_type,
                strategy,
                int(success),
                cost,
                episode_number,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read / aggregate
    # ------------------------------------------------------------------

    def get_stats(
        self,
        failure_type: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return aggregated attempt / success / success-rate per group.

        Optionally filter by ``failure_type`` and/or ``strategy``.
        The result has columns ``[failure_type, strategy, attempts,
        successes, success_rate]``.
        """
        query = "SELECT failure_type, strategy, success FROM strategy_outcomes WHERE 1=1"
        params: list = []
        if failure_type is not None:
            query += " AND failure_type = ?"
            params.append(failure_type)
        if strategy is not None:
            query += " AND strategy = ?"
            params.append(strategy)

        df = pd.read_sql_query(query, self._conn, params=params)
        if df.empty:
            return pd.DataFrame(
                columns=["failure_type", "strategy", "attempts", "successes", "success_rate"]
            )

        grouped = (
            df.groupby(["failure_type", "strategy"])
            .agg(attempts=("success", "size"), successes=("success", "sum"))
            .reset_index()
        )
        grouped["success_rate"] = grouped["successes"] / grouped["attempts"]
        return grouped

    def get_full_history(self) -> pd.DataFrame:
        """Return every row in the database as a DataFrame."""
        return pd.read_sql_query(
            "SELECT * FROM strategy_outcomes ORDER BY id", self._conn
        )

    # ------------------------------------------------------------------
    # Replay into selector
    # ------------------------------------------------------------------

    def load_into_selector(self, selector) -> None:  # noqa: ANN001 – avoids circular import
        """Replay all stored outcomes into an AdaptiveSelector.

        This lets a selector resume learning from where a previous run
        left off, rather than starting from scratch.
        """
        from src.failures.taxonomy import FailureType
        from src.recovery.strategies import RecoveryStrategy

        rows = self._conn.execute(
            "SELECT failure_type, strategy, success, cost "
            "FROM strategy_outcomes ORDER BY id"
        ).fetchall()
        for ft_str, st_str, success, cost in rows:
            ft = FailureType(ft_str)
            st = RecoveryStrategy(st_str)
            selector.update(ft, st, success=bool(success), cost=cost)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Delete all rows (for starting a fresh experiment run)."""
        self._conn.execute("DELETE FROM strategy_outcomes")
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
