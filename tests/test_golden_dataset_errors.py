import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.golden_dataset import load_golden_cases


def test_malformed_yaml_names_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("cases: [", encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(f"malformed YAML in {path}")):
        load_golden_cases(tmp_path)


def test_duplicate_ids_name_both_ids(tmp_path):
    (tmp_path / "a.yaml").write_text("- id: duplicate\n  input: one\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("- id: duplicate\n  input: two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate") as exc:
        load_golden_cases(tmp_path)
    assert "duplicate" in str(exc.value)


def test_invalid_case_names_file_and_entry(tmp_path):
    path = tmp_path / "invalid.yml"
    path.write_text("- id: missing-input\n", encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(f"invalid golden case in {path} (entry #0)")):
        load_golden_cases(tmp_path)
