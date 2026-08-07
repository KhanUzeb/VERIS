import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.targets import TargetError, my_agent_fn


def _stream_chunks(parts: list[str]):
    """Build a fake OpenAI streaming iterator from text fragments."""
    chunks = []
    for part in parts:
        delta = MagicMock()
        delta.content = part
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        chunks.append(chunk)
    return iter(chunks)


def _mock_client(stream_iter):
    client = MagicMock()
    client.chat.completions.create.return_value = stream_iter
    return client


@patch("veris.targets._client")
def test_streams_and_collects_final_message(mock_client_factory):
    mock_client_factory.return_value = (
        _mock_client(_stream_chunks(["Annual plans ", "refund in 30 days."])),
        "llama-3.3-70b-versatile",
    )
    out = my_agent_fn("What is the refund window for annual plans?")
    assert out == "Annual plans refund in 30 days."


@patch("veris.targets._client")
def test_retries_once_on_connection_error(mock_client_factory):
    request = MagicMock()
    failing = MagicMock()
    failing.chat.completions.create.side_effect = APIConnectionError(request=request)
    ok = _mock_client(_stream_chunks(["ok after retry"]))

    mock_client_factory.side_effect = [
        (failing, "m"),
        (ok, "m"),
    ]
    with patch("veris.targets.time.sleep"):
        assert my_agent_fn("hi") == "ok after retry"
    assert mock_client_factory.call_count == 2


@patch("veris.targets._client")
def test_retries_once_on_5xx(mock_client_factory):
    response = MagicMock()
    response.status_code = 503
    response.headers = {}
    err = APIStatusError("unavailable", response=response, body=None)

    failing = MagicMock()
    failing.chat.completions.create.side_effect = err
    ok = _mock_client(_stream_chunks(["recovered"]))

    mock_client_factory.side_effect = [(failing, "m"), (ok, "m")]
    with patch("veris.targets.time.sleep"):
        assert my_agent_fn("hi") == "recovered"


@patch("veris.targets._client")
def test_does_not_retry_on_4xx(mock_client_factory):
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    err = APIStatusError("bad request", response=response, body=None)

    client = MagicMock()
    client.chat.completions.create.side_effect = err
    mock_client_factory.return_value = (client, "m")

    with pytest.raises(TargetError, match="HTTP 400"):
        my_agent_fn("hi")
    assert mock_client_factory.call_count == 1


@patch("veris.targets._client")
def test_timeout_retries_then_fails(mock_client_factory):
    request = MagicMock()
    client = MagicMock()
    client.chat.completions.create.side_effect = APITimeoutError(request=request)
    mock_client_factory.return_value = (client, "m")

    with patch("veris.targets.time.sleep"), pytest.raises(TargetError, match="timed out"):
        my_agent_fn("hi")
    assert mock_client_factory.call_count == 2


@patch("veris.targets._resolve_endpoint")
def test_missing_key_fails_fast(mock_resolve):
    mock_resolve.side_effect = TargetError("no API key for TARGET_PROVIDER='groq'")
    with pytest.raises(TargetError, match="no API key"):
        my_agent_fn("hi")
