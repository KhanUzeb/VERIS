import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.langfuse_tracker import LangfuseTracker
from veris.schemas import CaseResult, RunReport


def _report():
    return RunReport(
        run_id="run-1",
        git_sha="abc123",
        case_results=[CaseResult(
            case_id="c1", input="question", actual_output="answer",
            scores={"faithfulness": 0.9, "toxicity": 0.98}, latency_ms=12.0,
            passed_min_score=True, regressed=True, regression_detail="drop",
        )],
    )


def test_tracker_without_keys_is_noop(monkeypatch):
    from veris import langfuse_tracker

    monkeypatch.setattr(langfuse_tracker, "SETTINGS", replace(langfuse_tracker.SETTINGS, langfuse_enabled=False))
    tracker = LangfuseTracker()
    assert tracker.enabled is False
    tracker.log_run(_report())


def test_tracker_v3_observation_path(monkeypatch):
    client = MagicMock()
    span = MagicMock()
    client.start_as_current_observation.return_value.__enter__.return_value = span
    tracker = object.__new__(LangfuseTracker)
    tracker.enabled = True
    tracker._client = client

    tracker.log_run(_report())

    client.start_as_current_observation.assert_called_once()
    kwargs = client.start_as_current_observation.call_args.kwargs
    assert kwargs["name"] == "eval:c1"
    assert kwargs["as_type"] == "span"
    span.update.assert_called_once_with(output={"actual_output": "answer"})
    assert span.score.call_count == 2
    client.flush.assert_called_once_with()


def test_tracker_legacy_trace_path():
    client = MagicMock(spec=["trace", "flush"])
    trace = client.trace.return_value
    tracker = object.__new__(LangfuseTracker)
    tracker.enabled = True
    tracker._client = client

    tracker.log_run(_report())

    client.trace.assert_called_once()
    kwargs = client.trace.call_args.kwargs
    assert kwargs["name"] == "eval:c1"
    assert kwargs["output"] == {"actual_output": "answer"}
    assert kwargs["tags"] == ["veris", "run-1", "regressed"]
    assert trace.score.call_count == 2
    client.flush.assert_called_once_with()


def test_tracker_swallows_flush_failure(capsys):
    client = MagicMock()
    client.flush.side_effect = RuntimeError("offline")
    tracker = object.__new__(LangfuseTracker)
    tracker.enabled = True
    tracker._client = client

    tracker.log_run(_report())

    assert "trace push failed" in capsys.readouterr().out


def test_tracker_initializes_v3_client(monkeypatch):
    from veris import langfuse_tracker

    client = MagicMock()
    fake_sdk = types.SimpleNamespace(Langfuse=MagicMock(return_value=client))
    monkeypatch.setitem(sys.modules, "langfuse", fake_sdk)
    monkeypatch.setattr(
        langfuse_tracker,
        "SETTINGS",
        replace(
            langfuse_tracker.SETTINGS,
            langfuse_enabled=True,
            langfuse_public_key="public",
            langfuse_secret_key="secret",
            langfuse_host="https://example.test",
        ),
    )

    tracker = LangfuseTracker()

    assert tracker.enabled is True
    assert tracker._client is client
    fake_sdk.Langfuse.assert_called_once_with(
        public_key="public",
        secret_key="secret",
        base_url="https://example.test",
    )


def test_tracker_falls_back_to_legacy_host(monkeypatch):
    from veris import langfuse_tracker

    client = MagicMock()

    def make_client(**kwargs):
        if "base_url" in kwargs:
            raise TypeError("legacy SDK")
        return client

    fake_sdk = types.SimpleNamespace(Langfuse=MagicMock(side_effect=make_client))
    monkeypatch.setitem(sys.modules, "langfuse", fake_sdk)
    monkeypatch.setattr(
        langfuse_tracker,
        "SETTINGS",
        replace(
            langfuse_tracker.SETTINGS,
            langfuse_enabled=True,
            langfuse_public_key="public",
            langfuse_secret_key="secret",
            langfuse_host="https://example.test",
        ),
    )

    tracker = LangfuseTracker()

    assert tracker.enabled is True
    assert tracker._client is client
    assert fake_sdk.Langfuse.call_args_list[-1].kwargs["host"] == "https://example.test"


def test_tracker_disables_when_sdk_init_fails(monkeypatch):
    from veris import langfuse_tracker

    fake_sdk = types.SimpleNamespace(Langfuse=MagicMock(side_effect=RuntimeError("offline")))
    monkeypatch.setitem(sys.modules, "langfuse", fake_sdk)
    monkeypatch.setattr(
        langfuse_tracker,
        "SETTINGS",
        replace(
            langfuse_tracker.SETTINGS,
            langfuse_enabled=True,
            langfuse_public_key="public",
            langfuse_secret_key="secret",
        ),
    )

    tracker = LangfuseTracker()

    assert tracker.enabled is False
    assert tracker._client is None
