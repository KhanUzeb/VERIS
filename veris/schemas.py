from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    """One golden test case. `expected_output` and `retrieval_context` are
    optional because not every metric needs them (e.g. toxicity only needs
    actual_output)."""
    id: str
    input: str
    expected_output: str | None = None
    retrieval_context: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class CaseResult(BaseModel):
    case_id: str
    input: str
    actual_output: str
    scores: dict[str, float]          # metric_name -> score (0-1)
    latency_ms: float
    passed_min_score: bool
    regressed: bool = False
    regression_detail: str | None = None
    error: str | None = None


class RunReport(BaseModel):
    run_id: str
    git_sha: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    case_results: list[CaseResult] = Field(default_factory=list)
    composite_score: float = 0.0
    passed: bool = False
    total_cases: int = 0
    failed_cases: int = 0
    regressed_cases: int = 0
    duration_s: float = 0.0
    flakiness_summary: str | None = None  # set when run with --repeat > 1

    def summarize(self) -> dict:
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "timestamp": self.timestamp.isoformat(),
            "composite_score": round(self.composite_score, 4),
            "passed": self.passed,
            "total_cases": self.total_cases,
            "failed_cases": self.failed_cases,
            "regressed_cases": self.regressed_cases,
            "duration_s": round(self.duration_s, 2),
        }
