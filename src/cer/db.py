from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> Database:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()
        return self

    def __exit__(self, *exc):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        assert self._conn is not None, "Database not opened. Use as context manager."
        return self._conn

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id          TEXT UNIQUE NOT NULL,
                commit_hash     TEXT NOT NULL,
                commit_short    TEXT NOT NULL,
                config_json     TEXT NOT NULL DEFAULT '{}',
                status          TEXT NOT NULL DEFAULT 'SUBMITTED',
                wandb_run_id    TEXT,
                wandb_url       TEXT,
                submitted_at    TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                cluster_host    TEXT NOT NULL,
                slurm_partition TEXT,
                error_message   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_commit ON experiments(commit_hash);
            CREATE INDEX IF NOT EXISTS idx_status ON experiments(status);
        """)

    def insert_experiment(
        self,
        job_id: str,
        commit_hash: str,
        config: dict,
        cluster_host: str,
        partition: str,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            """INSERT INTO experiments
               (job_id, commit_hash, commit_short, config_json, status,
                submitted_at, updated_at, cluster_host, slurm_partition)
               VALUES (?, ?, ?, ?, 'SUBMITTED', ?, ?, ?, ?)""",
            (
                job_id,
                commit_hash,
                commit_hash[:8],
                json.dumps(config),
                now,
                now,
                cluster_host,
                partition,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_experiment(self, job_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_status(
        self, job_id: str, status: str, error_message: str | None = None
    ):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE experiments
               SET status = ?, updated_at = ?, error_message = COALESCE(?, error_message)
               WHERE job_id = ?""",
            (status, now, error_message, job_id),
        )
        self.conn.commit()

    def update_wandb(self, job_id: str, run_id: str, url: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE experiments
               SET wandb_run_id = ?, wandb_url = ?, updated_at = ?
               WHERE job_id = ?""",
            (run_id, url, now, job_id),
        )
        self.conn.commit()

    def list_experiments(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM experiments ORDER BY submitted_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_active(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM experiments WHERE status IN ('SUBMITTED', 'PENDING', 'RUNNING')"
        ).fetchall()
        return [dict(r) for r in rows]
