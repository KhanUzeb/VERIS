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
import time

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import SETTINGS

# Caps concurrent judge-model calls across ALL GroqEvalModel instances,
# independent of the case-level thread pool. Without this, N worker
# threads × ~5 metrics/case can spike well past provider rate limits.
_JUDGE_CONCURRENCY = threading.Semaphore(SETTINGS.judge_concurrency)

# Process-wide running totals — read via `usage_totals()` for cost reporting.
_usage_lock = threading.Lock()
_usage_totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

# Rough $/1M tokens for cost estimates. Not billing-accurate, just enough
# to flag "this run cost noticeably more than usual" in reports.
_COST_PER_M_INPUT = 0.05
_COST_PER_M_OUTPUT = 0.08

# Judge replies are short structured scores — unbounded max_tokens made
# OpenRouter reserve up to 65k tokens and fail free-tier credit checks.
_JUDGE_MAX_TOKENS = 1024


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


def _is_rate_limited(err: BaseException) -> bool:
    text = str(err).lower()
    if "429" in text or "rate_limit" in text or "rate limit" in text:
        return True
    status = getattr(err, "status_code", None) or getattr(err, "code", None)
    if status == 429:
        return True
    body = getattr(err, "body", None)
    if isinstance(body, dict) and body.get("error", {}).get("code") == "rate_limit_exceeded":
        return True
    return False


def _is_retryable_primary(err: BaseException) -> bool:
    """Retry primary on rate limits / transient blips before spending fallback budget."""
    if _is_rate_limited(err):
        return True
    text = str(err).lower()
    return any(tok in text for tok in ("timeout", "temporarily", "503", "502", "overloaded"))


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

    @staticmethod
    def _message_text(resp) -> str:
        """Providers occasionally return empty choices / null content.
        Treat that as a hard failure so the circuit breaker can trip instead
        of feeding None into DeepEval metrics."""
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise RuntimeError("judge provider returned no choices")
        content = getattr(choices[0].message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("judge provider returned empty message content")
        return content

    def _call_groq(self, prompt: str) -> str:
        resp = self._groq().chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
        _record_usage(getattr(resp, "usage", None))
        return self._message_text(resp)

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
            max_tokens=_JUDGE_MAX_TOKENS,
        )
        _record_usage(getattr(resp, "usage", None))
        return self._message_text(resp)

    def _call_primary_with_backoff(self, primary, prompt: str):
        """Retry rate-limited primary calls in-process before fallback.

        Falling through to OpenRouter on every Groq 429 burns a dead credit
        balance and hides the real fix (wait / lower concurrency / smaller model).
        """
        attempts = 5
        delay = 2.0
        last_err: BaseException | None = None
        for i in range(attempts):
            try:
                return primary(prompt)
            except Exception as err:
                last_err = err
                if not _is_retryable_primary(err) or i == attempts - 1:
                    raise
                # Cap wait so a stuck provider doesn't hang a CI job forever.
                sleep_for = min(delay, 20.0)
                print(
                    f"[judge] rate-limited/transient error; "
                    f"retrying primary in {sleep_for:.1f}s "
                    f"({i + 1}/{attempts}): {err}"
                )
                time.sleep(sleep_for)
                delay *= 1.8
        assert last_err is not None
        raise last_err

    @retry(
        retry=retry_if_exception(_is_rate_limited),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=2, max=16),
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
                return self._call_primary_with_backoff(primary, prompt)
            except Exception as primary_err:
                if not fallback_key:
                    raise RuntimeError(
                        f"primary judge provider '{self.provider}' failed and no "
                        f"fallback API key is configured: {primary_err}"
                    ) from primary_err
                # Skip fallback when the primary error is purely rate-limit —
                # fallback usually won't help free-tier TPM, and a broke
                # OpenRouter key just adds noisy 402s.
                if _is_rate_limited(primary_err):
                    raise RuntimeError(
                        f"primary judge '{self.provider}' rate-limited after retries: "
                        f"{primary_err}. Lower EVAL_MAX_WORKERS / EVAL_JUDGE_CONCURRENCY, "
                        f"switch EVAL_JUDGE_MODEL to llama-3.3-70b-versatile, or wait a minute."
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
