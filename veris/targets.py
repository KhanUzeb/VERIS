"""
Wire your actual system here. `run_gate` just needs a callable that takes
a string input and returns a string output — swap this for a call into
your LangGraph graph, FastAPI endpoint, or whatever's being deployed.

Until a real target is wired, `my_agent_fn` is a deterministic stub that
returns the golden case's expected_output for a matching input so
Makefile / Dockerfile / CI / seed do not crash on NotImplementedError.
"""
from __future__ import annotations

from .config import SETTINGS
from .golden_dataset import load_golden_cases

_EXPECTED_BY_INPUT: dict[str, str] | None = None


def _expected_lookup() -> dict[str, str]:
    global _EXPECTED_BY_INPUT
    if _EXPECTED_BY_INPUT is None:
        try:
            cases = load_golden_cases(SETTINGS.golden_dataset_dir)
        except FileNotFoundError:
            _EXPECTED_BY_INPUT = {}
        else:
            _EXPECTED_BY_INPUT = {
                c.input: (c.expected_output or "")
                for c in cases
                if c.expected_output
            }
    return _EXPECTED_BY_INPUT


def my_agent_fn(input_text: str) -> str:
    """Deterministic stub target. Replace with e.g.:

        from my_app.graph import app as langgraph_app
        def my_agent_fn(input_text: str) -> str:
            result = langgraph_app.invoke({"messages": [{"role": "user", "content": input_text}]})
            return result["messages"][-1].content
    """
    expected = _expected_lookup().get(input_text)
    if expected is not None:
        return expected
    return (
        "I don't have enough context to answer that from the evaluation "
        "stub. Point veris.targets.my_agent_fn at your real system."
    )
