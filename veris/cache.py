"""
Content-addressable cache for metric scores. Keyed on a hash of
(case_id, case.input, case.expected_output, case.retrieval_context,
actual_output, judge model name, metrics fingerprint) — if none of those
changed since the last run, the judge call is skipped entirely and the
cached score is reused.

This matters because the judge model is an LLM call per metric per case.
At 100+ cases × 7 metrics, re-running the full suite every time for
cases whose target output didn't change is pure waste — both latency and
$ cost. Cache invalidates automatically the moment any input to the score
changes, including MetricSpec config edits via metrics_fingerprint().
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .config import SETTINGS, metrics_fingerprint
from .schemas import GoldenCase

_SCHEMA = """
CREATE TABLE IF NOT EXISTS score_cache (
    cache_key TEXT PRIMARY KEY,
    scores_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
);
"""


def cache_key(case: GoldenCase, actual_output: str, judge_model_name: str) -> str:
    payload = json.dumps(
        {
            "case_id": case.id,
            "input": case.input,
            "expected_output": case.expected_output,
            "retrieval_context": case.retrieval_context,
            "actual_output": actual_output,
            "judge": judge_model_name,
            "metrics_fp": metrics_fingerprint(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ScoreCache:
    def __init__(self, db_path: Path | None = None):
        default_path = SETTINGS.trend_db_path.with_name(
            SETTINGS.trend_db_path.stem + "_cache.db"
        )
        self.db_path = Path(db_path or default_path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def get(self, key: str) -> dict[str, float] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT scores_json FROM score_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            self.conn.execute(
                "UPDATE score_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (key,),
            )
            self.conn.commit()
        return json.loads(row[0])

    def set(self, key: str, scores: dict[str, float]) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO score_cache (cache_key, scores_json, created_at, hit_count)
                   VALUES (?, ?, ?, COALESCE((SELECT hit_count FROM score_cache WHERE cache_key = ?), 0))""",
                (key, json.dumps(scores), time.time(), key),
            )
            self.conn.commit()

    def stats(self) -> dict:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM score_cache"
            ).fetchone()
        return {"cached_entries": row[0], "total_hits": row[1]}

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM score_cache")
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()
