import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.targets import my_agent_fn


def test_stub_returns_expected_output_for_known_case():
    known_input = "What is the refund window for annual plans?"
    out = my_agent_fn(known_input)
    assert out == "Annual plans can be refunded within 30 days of purchase."


def test_stub_does_not_raise_on_unknown_input():
    out = my_agent_fn("this input is not in the golden dataset at all")
    assert isinstance(out, str)
    assert len(out) > 0
    assert "stub" in out.lower() or "real system" in out.lower()
