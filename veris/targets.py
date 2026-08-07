"""
Wire your actual system here. `run_gate` just needs a callable that takes
a string input and returns a string output.

`my_agent_fn` calls an OpenAI-compatible Chat Completions endpoint
(Groq / OpenRouter / OpenAI / custom base URL). Point TARGET_* env vars at
your LangGraph/FastAPI proxy or any chat-completions façade in front of the
system under test.

Resilience (required when runner uses ThreadPoolExecutor):
  - 30s per-call timeout (override with TARGET_TIMEOUT_S)
  - stream=True, tokens collected into the final message
  - retry once on transient failures (connection errors, timeouts, 5xx)
  - no retry on 4xx / malformed-request style errors
"""
from __future__ import annotations

import os
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

from .config import SETTINGS

# Per-case budget. A hung target must fail the case, not pin a worker forever.
_DEFAULT_TIMEOUT_S = 30.0
_MAX_ATTEMPTS = 2  # initial try + one retry on transient errors only


class TargetError(RuntimeError):
    """Raised when the target fails in a way that should fail the case."""


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _timeout_s() -> float:
    raw = _env("TARGET_TIMEOUT_S", str(_DEFAULT_TIMEOUT_S))
    try:
        return float(raw)
    except ValueError as e:
        raise TargetError(f"TARGET_TIMEOUT_S must be a number, got {raw!r}") from e


def _resolve_endpoint() -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for the OpenAI-compatible client."""
    provider = _env("TARGET_PROVIDER", "groq").lower()
    model = _env("TARGET_MODEL")
    api_key = _env("TARGET_API_KEY")
    base_url = _env("TARGET_BASE_URL")

    if provider == "groq":
        base_url = base_url or "https://api.groq.com/openai/v1"
        api_key = api_key or SETTINGS.groq_api_key
        model = model or "llama-3.3-70b-versatile"
    elif provider == "openrouter":
        base_url = base_url or "https://openrouter.ai/api/v1"
        api_key = api_key or SETTINGS.openrouter_api_key
        model = model or "meta-llama/llama-3.3-70b-instruct"
    elif provider == "openai":
        base_url = base_url or "https://api.openai.com/v1"
        api_key = api_key or _env("OPENAI_API_KEY")
        model = model or "gpt-4o-mini"
    elif provider == "custom":
        if not base_url:
            raise TargetError(
                "TARGET_PROVIDER=custom requires TARGET_BASE_URL "
                "(OpenAI-compatible chat completions base, e.g. "
                "http://localhost:8000/v1)"
            )
        model = model or "default"
    else:
        raise TargetError(
            f"unsupported TARGET_PROVIDER={provider!r}; "
            "expected groq|openrouter|openai|custom"
        )

    if not api_key and provider != "custom":
        raise TargetError(
            f"no API key for TARGET_PROVIDER={provider!r} — set TARGET_API_KEY "
            "or the provider's key (GROQ_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY)"
        )
    if not model:
        raise TargetError("TARGET_MODEL is required")

    return base_url, api_key, model


def _client() -> tuple[OpenAI, str]:
    base_url, api_key, model = _resolve_endpoint()
    # max_retries=0: we own the single transient retry below so 4xx never
    # silently burns a retry budget inside the SDK.
    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "not-needed",
        timeout=_timeout_s(),
        max_retries=0,
    )
    return client, model


def _is_transient(exc: BaseException) -> bool:
    """Connection blips / timeouts / 5xx — retry once. 4xx is a real bug."""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    # Some transports wrap resets as OSError / TimeoutError.
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return False


def _collect_stream(client: OpenAI, model: str, input_text: str) -> str:
    messages: list[dict[str, str]] = []
    system = _env("TARGET_SYSTEM_PROMPT")
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": input_text})

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(_env("TARGET_TEMPERATURE", "0")),
        stream=True,
    )
    parts: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if content:
            parts.append(content)
    text = "".join(parts).strip()
    if not text:
        raise TargetError("target returned an empty streamed completion")
    return text


def _call_once(input_text: str) -> str:
    client, model = _client()
    return _collect_stream(client, model, input_text)


def my_agent_fn(input_text: str) -> str:
    """Call the configured OpenAI-compatible target; collect streamed output."""
    last_err: BaseException | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return _call_once(input_text)
        except APIStatusError as e:
            # 4xx = real bug / bad request — fail the case immediately.
            if e.status_code < 500:
                raise TargetError(
                    f"target rejected request (HTTP {e.status_code}): {e.message}"
                ) from e
            last_err = e
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(0.5)
                continue
            raise TargetError(
                f"target failed after retry (HTTP {e.status_code}): {e.message}"
            ) from e
        except APITimeoutError as e:
            last_err = e
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(0.5)
                continue
            raise TargetError(
                f"target timed out after {_timeout_s():.0f}s (retried once)"
            ) from e
        except Exception as e:
            if _is_transient(e) and attempt + 1 < _MAX_ATTEMPTS:
                last_err = e
                time.sleep(0.5)
                continue
            raise TargetError(f"target call failed: {e}") from e
    raise TargetError(f"target failed after retry: {last_err}")
