"""Veris — local regression gate for LLM and agent systems.

Run a golden dataset against your target, score with DeepEval,
compare against a rolling baseline, and exit non-zero on regression.
Nothing runs automatically on commit — invoke the CLI when you want a check.
"""

__version__ = "2.1.1"
__all__ = ["__version__"]
