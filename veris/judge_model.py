"""
DeepEval-compatible judge LLM with two providers: **Groq** and **OpenRouter**.

Primary is `EVAL_JUDGE_PROVIDER` (`groq` | `openrouter`). Set both API keys to
enable failover. Each provider has its own model id:

- Groq → `EVAL_JUDGE_MODEL` (default `llama-3.3-70b-versatile`)
- OpenRouter → `EVAL_OPENROUTER_MODEL` (default `openrouter/free`, the free
  models router)

Fallback is used for daily quota / TPD, network errors, and empty replies.
Short TPM rate limits retry on the primary only.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Callable

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import SETTINGS

_JUDGE_CONCURRENCY = threading.Semaphore(SETTINGS.judge_concurrency)

_usage_lock = threading.Lock()
_usage_totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

# Rough $/1M tokens — enough to flag unusual run cost, not billing-accurate.
_COST_PER_M_INPUT = 0.05
_COST_PER_M_OUTPUT = 0.08

# Judge replies are short scores. Keep this low so OpenRouter free-tier
# credit checks (which reserve max_tokens) do not 402 mid-run.
_JUDGE_MAX_TOKENS = 256

_MAX_PROVIDER_WAIT_S = 180.0
_DEFAULT_BACKOFF_CAP_S = 20.0
_SUPPORTED_PROVIDERS = frozenset({"groq", "openrouter"})


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


def _redact_secrets(text: str) -> str:
    """Strip common API-key shapes from provider errors before logging."""
    text = re.sub(r"\b(gsk_|sk-or-v1-|sk-or-|sk-)[A-Za-z0-9_-]{8,}", r"\1***", text)
    text = re.sub(
        r"(api[_-]?key|authorization|bearer)\s*[:=]\s*\S+",
        r"\1: ***",
        text,
        flags=re.IGNORECASE,
    )
    return text


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


def _is_daily_quota(err: BaseException) -> bool:
    """Daily quotas (TPD/RPD) will not recover inside a CI backoff window."""
    text = str(err).lower()
    return any(
        tok in text
        for tok in (
            "tokens per day",
            "token per day",
            "tpd",
            "per day (tpd)",
            "daily rate limit",
            "requests per day",
            "rpd",
        )
    )


def _parse_retry_after_seconds(err: BaseException) -> float | None:
    """Extract a wait hint from Groq/OpenAI-style rate-limit messages."""
    text = str(err)
    match = re.search(
        r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60.0 + seconds
    response = getattr(err, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None) or {}
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    return None


def _backoff_seconds(err: BaseException, delay: float) -> float:
    hinted = _parse_retry_after_seconds(err)
    if hinted is not None and hinted > 0:
        return min(hinted, _MAX_PROVIDER_WAIT_S)
    return min(delay, _DEFAULT_BACKOFF_CAP_S)


def _is_retryable_primary(err: BaseException) -> bool:
    if _is_rate_limited(err):
        return not _is_daily_quota(err)
    text = str(err).lower()
    return any(tok in text for tok in ("timeout", "temporarily", "503", "502", "overloaded"))


def _deepeval_base() -> type:
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
        return DeepEvalBaseLLM
    except ImportError:
        return object


class GroqEvalModel(_deepeval_base()):  # type: ignore[misc]
    """Shared judge used by DeepEval metrics (name kept for compatibility)."""

    def __init__(self, model_name: str | None = None, provider: str | None = None):
        self.provider = (provider or SETTINGS.judge_provider).lower().strip()
        if self.provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"unsupported EVAL_JUDGE_PROVIDER={self.provider!r}; "
                f"expected one of {sorted(_SUPPORTED_PROVIDERS)}"
            )
        if model_name is not None:
            self.model_name = model_name
        elif self.provider == "openrouter":
            self.model_name = SETTINGS.openrouter_model
        else:
            self.model_name = SETTINGS.judge_model
        self._groq_client = None
        self._or_client = None

    def load_model(self):
        return self.model_name

    def _groq_model_id(self) -> str:
        return self.model_name if self.provider == "groq" else SETTINGS.judge_model

    def _openrouter_model_id(self) -> str:
        return self.model_name if self.provider == "openrouter" else SETTINGS.openrouter_model

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
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise RuntimeError("judge provider returned no choices")
        content = getattr(choices[0].message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("judge provider returned empty message content")
        return content

    def _call_groq(self, prompt: str) -> str:
        resp = self._groq().chat.completions.create(
            model=self._groq_model_id(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
        _record_usage(getattr(resp, "usage", None))
        return self._message_text(resp)

    def _call_openrouter(self, prompt: str) -> str:
        resp = self._openrouter().chat.completions.create(
            model=self._openrouter_model_id(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
        _record_usage(getattr(resp, "usage", None))
        return self._message_text(resp)

    def _provider_pair(
        self,
    ) -> tuple[Callable[[str], str], Callable[[str], str], str, str]:
        """Return (primary, fallback, fallback_key, fallback_name)."""
        if self.provider == "groq":
            return (
                self._call_groq,
                self._call_openrouter,
                SETTINGS.openrouter_api_key,
                "openrouter",
            )
        return (
            self._call_openrouter,
            self._call_groq,
            SETTINGS.groq_api_key,
            "groq",
        )

    def _call_primary_with_backoff(self, primary: Callable[[str], str], prompt: str) -> str:
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
                sleep_for = _backoff_seconds(err, delay)
                print(
                    f"[judge] rate-limited/transient error; "
                    f"retrying primary in {sleep_for:.1f}s "
                    f"({i + 1}/{attempts}): {_redact_secrets(str(err))}"
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
    def generate(self, prompt: str, schema=None, **_kwargs) -> str:
        # DeepEval 4.x calls generate(prompt, schema=Model). We accept the
        # extra kwargs so generate_with_schema does not TypeError-and-retry;
        # returning JSON text is valid — DeepEval's trimAndLoadJson parses it.
        del schema, _kwargs
        with _JUDGE_CONCURRENCY:
            primary, fallback, fallback_key, fallback_name = self._provider_pair()
            try:
                return self._call_primary_with_backoff(primary, prompt)
            except Exception as primary_err:
                can_failover = bool(fallback_key) and SETTINGS.judge_failover
                if not can_failover:
                    if not fallback_key:
                        raise RuntimeError(
                            f"primary judge provider '{self.provider}' failed and no "
                            f"fallback API key is configured: "
                            f"{_redact_secrets(str(primary_err))}"
                        ) from primary_err
                    raise RuntimeError(
                        f"primary judge provider '{self.provider}' failed "
                        f"({_redact_secrets(str(primary_err))}). "
                        f"Set EVAL_JUDGE_FAILOVER=1 to allow cross-provider fallback."
                    ) from primary_err
                # Short TPM: keep failing on primary so operators lower concurrency
                # or wait. Daily quota: fall through to the other provider.
                if _is_rate_limited(primary_err) and not _is_daily_quota(primary_err):
                    raise RuntimeError(
                        f"primary judge '{self.provider}' rate-limited after retries: "
                        f"{_redact_secrets(str(primary_err))}. "
                        f"Lower EVAL_MAX_WORKERS / EVAL_JUDGE_CONCURRENCY, "
                        f"or wait for the TPM window to refill."
                    ) from primary_err
                if _is_daily_quota(primary_err):
                    print(
                        f"[judge] primary '{self.provider}' hit a daily quota; "
                        f"trying fallback '{fallback_name}' "
                        f"(prompts leave the primary provider)"
                    )
                try:
                    return fallback(prompt)
                except Exception as fallback_err:
                    raise RuntimeError(
                        f"primary judge '{self.provider}' failed "
                        f"({_redact_secrets(str(primary_err))}); "
                        f"fallback '{fallback_name}' also failed "
                        f"({_redact_secrets(str(fallback_err))})"
                    ) from fallback_err

    async def a_generate(self, prompt: str, schema=None, **_kwargs) -> str:
        # Run the sync provider call in a worker thread so DeepEval's
        # async_mode path does not block the event loop.
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        """Identity used in reports + score-cache keys.

        When failover is enabled with both keys, include the peer model so
        cache entries do not silently mix primary-only and failover scores.
        """
        base = f"{self.provider}/{self.model_name}"
        if not SETTINGS.judge_failover:
            return base
        if self.provider == "groq" and SETTINGS.openrouter_api_key:
            return f"{base}|failover={SETTINGS.openrouter_model}"
        if self.provider == "openrouter" and SETTINGS.groq_api_key:
            return f"{base}|failover={SETTINGS.judge_model}"
        return base
