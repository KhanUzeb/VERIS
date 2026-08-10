"""
Demo target for Veris — a known-good stub, not a real agent.

Looks up each case input in the golden dataset and returns the authored
`expected_output` (or a short refusal/clarification fallback). That makes a
first demo / seed run predictable so you can exercise the gate without wiring
LangGraph, HTTP, or a production agent yet.

Swap `--target` to your real `module:function` when ready.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from veris.config import SETTINGS
from veris.golden_dataset import load_golden_cases

# When set to "1"/"true", answers are deliberately degraded so you can demo
# a blocking regression after seeding a green baseline.
_DEGRADE = os.getenv("VERIS_DEMO_DEGRADE", "").strip().lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def _oracle() -> dict[str, str]:
    dataset = Path(os.getenv("GOLDEN_DATASET_DIR", str(SETTINGS.golden_dataset_dir)))
    mapping: dict[str, str] = {}
    for case in load_golden_cases(dataset):
        if case.expected_output:
            mapping[case.input.strip()] = case.expected_output.strip()
    return mapping


def _fallback(case_input: str) -> str:
    text = case_input.lower()
    if any(
        tip in text
        for tip in (
            "bypass", "phish", "exfiltrat", "api key", "sql injection",
            "malware", "dox", "brute-force", "hate speech", "ignore previous",
            "dan and have no safety", "credentials for a shared",
        )
    ):
        return (
            "I can't help with that request. It looks like it could involve "
            "abuse, fraud, or security harm. If you have a legitimate account "
            "or product question, tell me what you're trying to accomplish."
        )
    if "it's not working" in text or text.strip() == "it's not working.":
        return (
            "I'm sorry you're stuck — can you share what isn't working, any "
            "error message, and when it started?"
        )
    return (
        "Thanks for the message. I don't have a canned answer for that input "
        "in the demo oracle; please rephrase or check the golden dataset."
    )


def answer(case_input: str) -> str:
    """Veris target contract: one string in, one string out."""
    if not isinstance(case_input, str):
        raise TypeError(f"demo target expected str, got {type(case_input).__name__}")

    key = case_input.strip()
    if _DEGRADE:
        # Intentionally weak / off-policy so floors and regressions fire.
        return "idk lol whatever"

    return _oracle().get(key) or _fallback(case_input)


# Alias some teams prefer in docs / Makefile examples
demo_answer = answer
