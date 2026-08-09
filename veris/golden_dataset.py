"""Loads golden test cases from a directory of YAML files. Scales to 100+
cases by just adding more files — nothing here assumes a single file."""
from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import GoldenCase


def load_golden_cases(directory: Path, tags: list[str] | None = None) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"golden dataset dir not found: {directory}")

    for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        try:
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or []
        except yaml.YAMLError as e:
            raise ValueError(f"malformed YAML in {path}: {e}") from e

        if isinstance(raw, dict):
            if "cases" not in raw:
                raise ValueError(
                    f"invalid golden dataset in {path}: top-level mapping must contain 'cases'"
                )
            raw = raw["cases"]
        if not isinstance(raw, list):
            raise ValueError(
                f"invalid golden dataset in {path}: top-level value must be a list or a mapping with 'cases'"
            )
        for i, entry in enumerate(raw):
            try:
                case = GoldenCase(**entry)
            except Exception as e:
                raise ValueError(f"invalid golden case in {path} (entry #{i}): {e}") from e
            cases.append(case)

    if tags:
        wanted = set(tags)
        cases = [c for c in cases if wanted & set(c.tags)]

    _validate_unique_ids(cases)
    return cases


def _validate_unique_ids(cases: list[GoldenCase]) -> None:
    seen = set()
    dupes = set()
    for c in cases:
        if c.id in seen:
            dupes.add(c.id)
        seen.add(c.id)
    if dupes:
        raise ValueError(f"duplicate golden case ids across files: {sorted(dupes)}")
