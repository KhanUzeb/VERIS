import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_runtime_dependencies_live_under_project_not_urls():
    import tomllib

    data = tomllib.loads(
        Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text(encoding="utf-8")
    )
    deps = data["project"]["dependencies"]
    assert any(item.startswith("deepeval") for item in deps)
    assert "click" in " ".join(deps)
    assert "dependencies" not in data["project"]["urls"]
