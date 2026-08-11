import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tenacity import wait_none

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.judge_model import GroqEvalModel


def test_judge_connection_error_is_not_wrapped_in_retry_error(monkeypatch):
    from dataclasses import replace

    from veris.judge_model import SETTINGS

    monkeypatch.setattr(
        "veris.judge_model.SETTINGS",
        replace(SETTINGS, judge_failover=True, openrouter_api_key="test-or-key"),
    )
    model = GroqEvalModel(model_name="test-model", provider="groq")
    model.generate.retry.wait = wait_none()
    with patch.object(model, "_call_groq", side_effect=RuntimeError("network blocked")), \
         patch.object(model, "_call_openrouter", side_effect=RuntimeError("fallback blocked")):
        with pytest.raises(RuntimeError, match="also failed") as exc:
            model.generate("hello")
    assert "RetryError" not in str(exc.value)


def test_primary_openrouter_uses_configured_model():
    model = GroqEvalModel(model_name="provider/custom-model", provider="openrouter")
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="ok"))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    model._or_client = client

    assert model._call_openrouter("hello") == "ok"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "provider/custom-model"
    assert kwargs["max_tokens"] == 256


def test_groq_primary_falls_back_to_openrouter_free_model(monkeypatch):
    from dataclasses import replace

    from veris.judge_model import SETTINGS

    monkeypatch.setattr(
        "veris.judge_model.SETTINGS",
        replace(
            SETTINGS,
            judge_provider="groq",
            judge_model="llama-3.3-70b-versatile",
            openrouter_model="openrouter/free",
            openrouter_api_key="test-or-key",
        ),
    )

    model = GroqEvalModel(provider="groq")
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="ok"))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    model._or_client = client

    assert model._call_openrouter("hello") == "ok"
    assert client.chat.completions.create.call_args.kwargs["model"] == "openrouter/free"


def test_rate_limit_retries_primary_without_fallback():
    model = GroqEvalModel(model_name="test-model", provider="groq")
    model.generate.retry.wait = wait_none()
    calls = {"n": 0}

    def flaky(_prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")
        return "ok"

    with patch.object(model, "_call_groq", side_effect=flaky), \
         patch.object(model, "_call_openrouter") as fallback, \
         patch("veris.judge_model.time.sleep"):
        assert model.generate("hello") == "ok"
    assert calls["n"] == 3
    fallback.assert_not_called()


def test_retry_after_hint_is_honored():
    from veris.judge_model import _backoff_seconds, _parse_retry_after_seconds

    err = RuntimeError(
        "Error code: 429 - Please try again in 2m52.8s. Need more tokens?"
    )
    assert _parse_retry_after_seconds(err) == pytest.approx(172.8)
    assert _backoff_seconds(err, delay=2.0) == pytest.approx(172.8)


def test_daily_quota_falls_back_without_primary_retries(monkeypatch):
    from dataclasses import replace

    from veris.judge_model import SETTINGS

    monkeypatch.setattr(
        "veris.judge_model.SETTINGS",
        replace(
            SETTINGS,
            judge_failover=True,
            openrouter_api_key="test-or-key",
            openrouter_model="openrouter/free",
        ),
    )

    model = GroqEvalModel(model_name="test-model", provider="groq")
    model.generate.retry.wait = wait_none()
    primary_calls = {"n": 0}

    def tpd(_prompt):
        primary_calls["n"] += 1
        raise RuntimeError(
            "Error code: 429 - Rate limit reached for model `llama-3.3-70b-versatile` "
            "on tokens per day (TPD): Limit 100000, Used 99792, Requested 408. "
            "Please try again in 2m52.8s."
        )

    with patch.object(model, "_call_groq", side_effect=tpd), \
         patch.object(model, "_call_openrouter", return_value="ok") as fallback, \
         patch("veris.judge_model.time.sleep") as sleep:
        assert model.generate("hello") == "ok"
    assert primary_calls["n"] == 1
    fallback.assert_called_once_with("hello")
    sleep.assert_not_called()


def test_redact_secrets_strips_key_shapes():
    from veris.judge_model import _redact_secrets

    raw = (
        "auth failed api_key=gsk_abcdefghijklmnopqrstuvwxyz "
        "Authorization: Bearer sk-or-v1-abcdef123456 "
        "token sk-or-v1-zzzzzzzzzzzz"
    )
    cleaned = _redact_secrets(raw)
    assert "abcdefghijklmnopqrstuvwxyz" not in cleaned
    assert "abcdef123456" not in cleaned
    assert "zzzzzzzzzzzz" not in cleaned
    assert "***" in cleaned


def test_failover_disabled_does_not_call_fallback(monkeypatch):
    from dataclasses import replace

    from veris.judge_model import SETTINGS

    monkeypatch.setattr(
        "veris.judge_model.SETTINGS",
        replace(SETTINGS, judge_failover=False, openrouter_api_key="test-or-key"),
    )
    model = GroqEvalModel(model_name="test-model", provider="groq")
    model.generate.retry.wait = wait_none()

    with patch.object(model, "_call_groq", side_effect=RuntimeError("network blocked")), \
         patch.object(model, "_call_openrouter") as fallback:
        with pytest.raises(RuntimeError, match="EVAL_JUDGE_FAILOVER"):
            model.generate("hello")
    fallback.assert_not_called()


def test_groq_call_sets_max_tokens():
    model = GroqEvalModel(model_name="llama-3.3-70b-versatile", provider="groq")
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="score"))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    model._groq_client = client
    assert model._call_groq("p") == "score"
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 256
