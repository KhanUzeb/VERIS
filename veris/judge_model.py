"""
Custom judge LLM used by DeepEval's metrics. Same pattern as the
GroqEvalModel in agentic-evals v1, extended with an OpenRouter fallback
so a single Groq rate-limit blip doesn't fail the whole CI run.

Also owns two things a bare wrapper wouldn't: a process-wide concurrency
semaphore (the runner's ThreadPoolExecutor fans out per-case, but each
case fires several metric calls, so real concurrent judge calls can be a
multiple of max_workers — this caps it independently) and running token
totals, so a CI run can report what it cost.

Primary provider is selected via EVAL_JUDGE_PROVIDER (groq|openrouter).
The other provider is used as fallback only when its API key is set.

DeepEval is imported lazily so CLI commands that never score (validate,
list-cases, trend, dry-run) and unit tests that mock the judge don't
require deepeval to be installed just to import the package.
"""
from __future__ import annotations

import threading

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import SETTINGS

# Caps concurrent judge-model calls across ALL GroqEvalModel instances,
# independent of the case-level thread pool. Without this, N worker
# threads × ~5 metrics/case can spike well past provider rate limits.
_JUDGE_CONCURRENCY = threading.Semaphore(6)

# Process-wide running totals — read via `usage_totals()` for cost reporting.
_usage_lock = threading.Lock()
_usage_totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

# Rough $/1M tokens for cost estimates. Not billing-accurate, just enough
# to flag "this run cost noticeably more than usual" in reports.
_COST_PER_M_INPUT = 0.05
_COST_PER_M_OUTPUT = 0.08


def usage_totals() -> dict:
    with _usage_lock:
        totals: dict[str, int | float] = dict(_usage_totals)
    est_cost = (
        totals["prompt_tokens"] / 1_000_000 * _COST_PER_M_INPUT
        + totals["completion_tokens"] / 1_000_000 * _COST_PER_M_OUTPUT
    )
    totals["estimated_cost_usd"] = round(est_cost, 4)
    return totals


def reset_usage_totals() -> None:
    with _usage_lock:
        _usage_totals.update({"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})


def _record_usage(usage) -> None:
    if usage is None:
        return
    with _usage_lock:
        _usage_totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _usage_totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        _usage_totals["calls"] += 1


def _deepeval_base() -> type:
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
        return DeepEvalBaseLLM
    except ImportError:
        # Unit-test / CLI-only environments without deepeval installed.
        return object


class GroqEvalModel(_deepeval_base()):
    """DeepEval-compatible judge model with configurable primary provider
    and keyed OpenRouter/Groq fallback."""

    def __init__(self, model_name: str | None = None, provider: str | None = None):
        self.model_name = model_name or SETTINGS.judge_model
        self.provider = (provider or SETTINGS.judge_provider).lower().strip()
        if self.provider not in ("groq", "openrouter"):
            raise ValueError(
                f"unsupported EVAL_JUDGE_PROVIDER={self.provider!r}; "
                "expected 'groq' or 'openrouter'"
            )
        self._groq_client = None
        self._or_client = None

    def load_model(self):
        return self.model_name

    def _groq(self):
        if not SETTINGS.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set — cannot call Groq as judge provider"
            )
        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq(api_key=SETTINGS.groq_api_key)
        return self._groq_client

    def _openrouter(self):
        if not SETTINGS.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set — cannot call OpenRouter as judge provider"
            )
        if self._or_client is None:
            from openai import OpenAI
            self._or_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=SETTINGS.openrouter_api_key,
            )
        return self._or_client

    def _call_groq(self, prompt: str) -> str:
        resp = self._groq().chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        _record_usage(getattr(resp, "usage", None))
        return resp.choices[0].message.content

    def _call_openrouter(self, prompt: str) -> str:
        resp = self._openrouter().chat.completions.create(
            # A primary OpenRouter run must honor EVAL_JUDGE_MODEL. Keep the
            # known-good OpenRouter model for Groq-primary fallback, because
            # Groq model ids are not necessarily valid OpenRouter ids.
            model=(
                self.model_name
                if self.provider == "openrouter"
                else "meta-llama/llama-3.3-70b-instruct"
            ),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        _record_usage(getattr(resp, "usage", None))
        return resp.choices[0].message.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def generate(self, prompt: str) -> str:
        with _JUDGE_CONCURRENCY:
            primary = self._call_groq if self.provider == "groq" else self._call_openrouter
            fallback = self._call_openrouter if self.provider == "groq" else self._call_groq
            fallback_key = (
                SETTINGS.openrouter_api_key if self.provider == "groq" else SETTINGS.groq_api_key
            )
            try:
                return primary(prompt)
            except Exception as primary_err:
                if not fallback_key:
                    raise RuntimeError(
                        f"primary judge provider '{self.provider}' failed and no "
                        f"fallback API key is configured: {primary_err}"
                    ) from primary_err
                try:
                    return fallback(prompt)
                except Exception as fallback_err:
                    raise RuntimeError(
                        f"primary judge '{self.provider}' failed ({primary_err}); "
                        f"fallback also failed ({fallback_err})"
                    ) from fallback_err

    async def a_generate(self, prompt: str) -> str:
        # DeepEval calls this in async contexts; keep it simple and sync-backed
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"{self.provider}/{self.model_name}"
