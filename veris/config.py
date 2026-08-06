"""
Every tunable knob for the harness lives here. Nothing threshold-related
should be hardcoded elsewhere — the gate should be able to tighten over
time by editing this file alone.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from .schemas import GoldenCase


@dataclass(frozen=True)
class MetricSpec:
    name: str
    blocking: bool          # if True, a regression on this metric fails the gate
    min_score: float         # absolute floor, regardless of trend
    weight: float = 1.0      # used for the composite quality score in trend reports
    stdev_band_override: float | None = None  # per-metric noise tolerance; None = use SETTINGS.regression_stdev_band.
    # Set this when a metric is known to be judge-noisy (confirmed via
    # `run --repeat 5`) — widen the band so noise doesn't get mistaken for
    # a regression, WITHOUT touching min_score, which stays the real floor.
    # Caveat: when baseline stdev == 0 (identical historical scores), the
    # band and any override are ignored; the regression floor is hardcoded
    # to 0.9 * median in regression_gate.evaluate_case.
    requires_retrieval_context: bool = False
    # When True, the metric is undefined without retrieval_context and must
    # not be treated as a blocking "missing" failure for context-free cases.
    requires_expected_output: bool = False

    def is_applicable(self, case: GoldenCase) -> bool:
        if self.requires_retrieval_context and not case.retrieval_context:
            return False
        if self.requires_expected_output and not case.expected_output:
            return False
        return True


@dataclass(frozen=True)
class Settings:
    # --- Providers ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    judge_model: str = os.getenv("EVAL_JUDGE_MODEL", "llama-3.3-70b-versatile")
    # "groq" (default) | "openrouter" — primary provider; the other is used
    # as fallback when its API key is also configured.
    judge_provider: str = os.getenv("EVAL_JUDGE_PROVIDER", "groq")

    # --- Langfuse ---
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    langfuse_enabled: bool = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

    # --- Storage ---
    trend_db_path: Path = Path(os.getenv("TREND_DB_PATH", "./veris_trends.db"))
    golden_dataset_dir: Path = Path(os.getenv("GOLDEN_DATASET_DIR", "./golden_dataset"))

    # --- Regression gate ---
    baseline_window: int = 10          # how many past green runs form the rolling baseline
    regression_stdev_band: float = 1.5  # default noise band; overridable per-metric via MetricSpec.stdev_band_override
    min_baseline_samples: int = 3       # below this, fall back to absolute min_score only
    max_workers: int = 8                # concurrency for running cases against the target
    max_consecutive_judge_failures: int = 5  # circuit breaker: abort scoring after this many
    # judge-call failures in a row (almost always means a broken API key or
    # dead endpoint, not per-case flakiness) rather than silently burning
    # through the rest of the budget with every remaining case scored as 0.

    # --- Metrics (blocking = fails the run, advisory = reported but doesn't block) ---
    metrics: tuple = field(default_factory=lambda: (
        MetricSpec(
            "faithfulness", blocking=True, min_score=0.75, weight=1.5,
            requires_retrieval_context=True,
        ),
        MetricSpec("answer_relevancy", blocking=True, min_score=0.70, weight=1.0),
        MetricSpec(
            "hallucination", blocking=True, min_score=0.80, weight=1.5,
            requires_retrieval_context=True,
        ),  # inverted: higher = less hallucination
        MetricSpec(
            "context_precision", blocking=False, min_score=0.60, weight=1.0,
            requires_retrieval_context=True,
        ),
        MetricSpec(
            "context_recall", blocking=False, min_score=0.60, weight=1.0,
            requires_retrieval_context=True,
        ),
        MetricSpec(
            "answer_correctness", blocking=True, min_score=0.65, weight=1.2,
            requires_retrieval_context=True, requires_expected_output=True,
        ),
        MetricSpec("toxicity", blocking=True, min_score=0.90, weight=1.0),  # inverted
    ))


SETTINGS = Settings()


def metrics_fingerprint() -> str:
    """Stable hash of metric names + floors + applicability flags.
    Included in the score-cache key so changing config.py invalidates stale
    cached scores instead of silently reusing them."""
    payload = [
        {
            "name": m.name,
            "blocking": m.blocking,
            "min_score": m.min_score,
            "weight": m.weight,
            "stdev_band_override": m.stdev_band_override,
            "requires_retrieval_context": m.requires_retrieval_context,
            "requires_expected_output": m.requires_expected_output,
        }
        for m in SETTINGS.metrics
    ]
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
