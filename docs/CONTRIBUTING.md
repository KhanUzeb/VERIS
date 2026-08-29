# Contributing to Veris

Thanks for helping improve Veris, the local, agent-agnostic LLM evaluation and
regression testing gate. Start with the [README](../README.md) for the
big picture.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env   # add GROQ_API_KEY for live runs
pre-commit install       # one-time setup for automatic lint/type checks
```

## Checks

```bash
make test       # pytest + coverage
make lint       # ruff
make typecheck  # mypy
make validate   # golden dataset schema (no LLM)
make dry-run    # plan cases without judging
```

Veris is intentionally agent-agnostic. Do not add a sample agent or provider
adapter to the harness package. Add integration coverage through a test-only
callable or a consuming project that supplies `--target module:function`.

## Guidelines

- Keep gate thresholds in `veris/config.py`. Do not hardcode floors in the runner.
- Prefer tests that avoid live LLM calls (monkeypatch `runner.score_deepeval`).
- Metric applicability belongs on `MetricSpec` (`requires_retrieval_context`, etc.).
- User-facing product name is **Veris**; Python package / CLI is `veris`.
- Do not commit `.env`, trend/cache DBs, or report artifacts.

## Adding a golden case

1. Add an entry under `golden_dataset/*.yaml` with a unique `id`.
2. Run `veris validate` and `veris list-cases`.
3. Prefer tags so you can run focused subsets (`--tags …`).

## Adding a metric

1. Implement scoring in `veris/metrics/`.
2. Register a `MetricSpec` in `veris/config.py` (blocking, floor, weight, applicability).
3. Cache keys include `metrics_fingerprint()`. No manual cache clear is required after config changes, but `--no-cache` is fine while iterating.
4. Add a unit test for gate behavior (especially N/A / missing / floor).

## Releases

Version is set in `veris/__init__.py` and `pyproject.toml`. Keep them in sync.
