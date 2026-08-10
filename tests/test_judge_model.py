import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tenacity import wait_none

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.judge_model import GroqEvalModel


def test_judge_connection_error_is_not_wrapped_in_retry_error():
    model = GroqEvalModel(model_name="test-model", provider="groq")
    model.generate.retry.wait = wait_none()
    with patch.object(model, "_call_groq", side_effect=RuntimeError("network blocked")), \
         patch.object(model, "_call_openrouter", side_effect=RuntimeError("fallback blocked")):
        with pytest.raises(RuntimeError, match="fallback also failed") as exc:
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
    assert kwargs["max_tokens"] == 1024


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


def test_groq_call_sets_max_tokens():
    model = GroqEvalModel(model_name="llama-3.3-70b-versatile", provider="groq")
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="score"))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    model._groq_client = client
    assert model._call_groq("p") == "score"
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 1024
