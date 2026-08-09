"""SQLite-backed history of every run + per-case score, so the regression
gate has something to compare against and `report trend` has data to plot.

Thread safety: runner.py hits this from inside a ThreadPoolExecutor (one
call per case, up to max_workers concurrent). A bare sqlite3.Connection is
NOT safe to share across threads by default — the original version of this
file did exactly that and would intermittently throw
"SQLiteObject created in a thread can only be used in that same thread" or
silently corrupt writes under load. Fixed by opening with
check_same_thread=False, running in WAL mode (readers don't block the
writer), and guarding every statement with a single re-entrant lock. This
serializes writes but that's fine — writes are microseconds vs the seconds
of LLM-judge latency per case, so it's not the bottleneck."""
from __future__ import annotations

import sqlite3
import statistics
import threading
from pathlib import Path

from .config import SETTINGS
from .schemas import RunReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    git_sha TEXT,
    timestamp TEXT,
    composite_score REAL,
    passed INTEGER,
    total_cases INTEGER,
    failed_cases INTEGER,
    regressed_cases INTEGER,
    duration_s REAL
);

CREATE TABLE IF NOT EXISTS case_scores (
    run_id TEXT,
    case_id TEXT,
    metric_name TEXT,
    score REAL,
    passed_run INTEGER,   -- per-case passed_min_score on incremental writes; baseline filters on runs.passed, not this column
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_case_metric ON case_scores(case_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_run_timestamp ON runs(timestamp);

CREATE TABLE IF NOT EXISTS judge_usage (
    run_id TEXT PRIMARY KEY,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    calls INTEGER,
    estimated_cost_usd REAL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
"""


class TrendStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or SETTINGS.trend_db_path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def begin_run(self, run_id: str, git_sha: str | None, total_cases: int) -> None:
        """Insert a stub parent `runs` row before incremental case_scores
        writes so baselines can JOIN completed scores if the process dies
        mid-run. Finalized later via save_run (INSERT OR REPLACE)."""
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, git_sha, timestamp, composite_score, passed,
                    total_cases, failed_cases, regressed_cases, duration_s)
                   VALUES (?,?,datetime('now'),0.0,0,?,?,?,0.0)""",
                (run_id, git_sha, total_cases, 0, 0),
            )
            self.conn.commit()

    def save_run(self, report: RunReport, write_case_scores: bool = True) -> None:
        """Writes the parent `runs` row. If the caller already streamed
        per-case scores via save_case_score during the run (runner.py does
        this so a crash mid-run doesn't lose completed cases), pass
        write_case_scores=False to avoid inserting duplicates."""
        with self._lock:
            c = self.conn.cursor()
            c.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, git_sha, timestamp, composite_score, passed,
                    total_cases, failed_cases, regressed_cases, duration_s)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    report.run_id, report.git_sha, report.timestamp.isoformat(),
                    report.composite_score, int(report.passed), report.total_cases,
                    report.failed_cases, report.regressed_cases, report.duration_s,
                ),
            )
            if write_case_scores:
                for cr in report.case_results:
                    for metric_name, score in cr.scores.items():
                        c.execute(
                            """INSERT INTO case_scores
                               (run_id, case_id, metric_name, score, passed_run)
                               VALUES (?,?,?,?,?)""",
                            (report.run_id, cr.case_id, metric_name, score, int(report.passed)),
                        )
            self.conn.commit()

    def save_case_score(self, run_id: str, case_id: str, metric_name: str,
                         score: float, passed_run: bool) -> None:
        """Incremental write used by the runner as each case finishes, so a
        crash mid-run doesn't lose every case that already completed."""
        with self._lock:
            self.conn.execute(
                """INSERT INTO case_scores (run_id, case_id, metric_name, score, passed_run)
                   VALUES (?,?,?,?,?)""",
                (run_id, case_id, metric_name, score, int(passed_run)),
            )
            self.conn.commit()

    def baseline_for_case_metric(
        self, case_id: str, metric_name: str, window: int, green_only: bool = True
    ) -> tuple[float, float, int] | None:
        """Returns (median, stdev, sample_count) from the last `window` runs
        for this case+metric, restricted to green (passed) runs by default
        so a regression doesn't get "baked in" as the new normal."""
        query = """
            SELECT cs.score FROM case_scores cs
            JOIN runs r ON cs.run_id = r.run_id
            WHERE cs.case_id = ? AND cs.metric_name = ?
        """
        params: list = [case_id, metric_name]
        if green_only:
            query += " AND r.passed = 1"
        query += " ORDER BY r.timestamp DESC LIMIT ?"
        params.append(window)

        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        scores = [r[0] for r in rows]
        if len(scores) < 1:
            return None
        median = statistics.median(scores)
        stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        return median, stdev, len(scores)

    def recent_runs(self, limit: int = 30) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT run_id, git_sha, timestamp, composite_score, passed,
                          total_cases, failed_cases, regressed_cases, duration_s
                   FROM runs ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        cols = ["run_id", "git_sha", "timestamp", "composite_score", "passed",
                "total_cases", "failed_cases", "regressed_cases", "duration_s"]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                """SELECT run_id, git_sha, timestamp, composite_score, passed,
                          total_cases, failed_cases, regressed_cases, duration_s
                   FROM runs WHERE run_id = ?""",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        cols = ["run_id", "git_sha", "timestamp", "composite_score", "passed",
                "total_cases", "failed_cases", "regressed_cases", "duration_s"]
        return dict(zip(cols, row, strict=True))

    def get_case_scores(self, run_id: str) -> dict[str, dict[str, float]]:
        """case_id -> {metric_name: score} for one run. Used by `diff`."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT case_id, metric_name, score FROM case_scores WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for case_id, metric_name, score in rows:
            out.setdefault(case_id, {})[metric_name] = score
        return out

    def resolve_run_id(self, ref: str) -> str | None:
        """Accepts a run_id, git sha, or 'latest'/'previous'. Returns the
        matching run_id or None."""
        with self._lock:
            if ref == "latest":
                row = self.conn.execute(
                    "SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                return row[0] if row else None
            if ref == "previous":
                row = self.conn.execute(
                    "SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT 1 OFFSET 1"
                ).fetchone()
                return row[0] if row else None
            row = self.conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ? OR git_sha = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (ref, ref),
            ).fetchone()
            return row[0] if row else None

    def save_usage(self, run_id: str, usage: dict) -> None:
        """Persists judge_model.usage_totals() for a run. Without this,
        cost data only lives in-process and is gone the moment the CI
        job exits — which defeats the point of tracking it at all."""
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO judge_usage
                   (run_id, prompt_tokens, completion_tokens, calls, estimated_cost_usd)
                   VALUES (?,?,?,?,?)""",
                (run_id, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                 usage.get("calls", 0), usage.get("estimated_cost_usd", 0.0)),
            )
            self.conn.commit()

    def usage_history(self, limit: int = 30) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT ju.run_id, r.timestamp, r.git_sha, ju.calls,
                          ju.prompt_tokens, ju.completion_tokens, ju.estimated_cost_usd
                   FROM judge_usage ju JOIN runs r ON ju.run_id = r.run_id
                   ORDER BY r.timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        cols = ["run_id", "timestamp", "git_sha", "calls", "prompt_tokens",
                "completion_tokens", "estimated_cost_usd"]
        return [dict(zip(cols, r, strict=True)) for r in rows]

    def prune_older_than(self, days: int) -> int:
        """Deletes runs (and their case_scores/judge_usage) older than N
        days. The trend store grows unbounded otherwise — fine for a
        while, but worth having before it's a problem."""
        with self._lock:
            cutoff_rows = self.conn.execute(
                "SELECT run_id FROM runs WHERE timestamp < datetime('now', ?)",
                (f"-{days} days",),
            ).fetchall()
            run_ids = [r[0] for r in cutoff_rows]
            if not run_ids:
                return 0
            placeholders = ",".join("?" * len(run_ids))
            self.conn.execute(f"DELETE FROM case_scores WHERE run_id IN ({placeholders})", run_ids)
            self.conn.execute(f"DELETE FROM judge_usage WHERE run_id IN ({placeholders})", run_ids)
            self.conn.execute(f"DELETE FROM runs WHERE run_id IN ({placeholders})", run_ids)
            self.conn.commit()
        return len(run_ids)

    def close(self):
        with self._lock:
            self.conn.close()
