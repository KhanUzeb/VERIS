# Veris

A local regression gate for LLM agents, RAG pipelines, and plain Python callables.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![Docs](https://img.shields.io/badge/docs-user%20guide-informational)](docs/USER_GUIDE.md)

Veris runs a golden dataset against your function, scores the outputs with DeepEval, compares them to a rolling SQLite baseline, and exits `0` or `1`. That's it. No bundled agent, no hosted dashboard, no commit hooks.

It answers three questions:

1. Did quality drop since the last known-good run?
2. Is that a real regression or just judge-model noise?
3. What does the trend look like?

## Features

- Works with any `Callable[[str], str]`: LangGraph, FastAPI clients, RAG pipelines, plain functions.
- DeepEval metrics: faithfulness, answer relevancy, hallucination, toxicity.
- Regression detection via absolute floors plus a rolling median and standard-deviation band over recent green runs.
- Noise handling through configurable bands and optional flakiness audits (`--repeat N`).
- Fully local: SQLite trend store, content-addressed score cache, no cloud dependency.
- Free-tier friendly: Groq and OpenRouter judges with automatic failover, plus a circuit breaker that aborts a broken run before it burns tokens.
- CI-ready exit codes.

## Quickstart

```bash
pip install -e .
cp .env.example .env        # set GROQ_API_KEY; OPENROUTER_API_KEY is an optional fallback

# seed a baseline once, against a build you trust
python scripts/seed_baseline.py --target your_module:your_callable --n 3

# preview the run (zero LLM calls)
veris run --target your_module:your_callable --dry-run

# run the gate
veris run --target your_module:your_callable
```

## Your target

A function that takes the case input and returns the output, both strings:

```python
def answer(case_input: str) -> str:
    return my_agent.invoke(case_input)
```

Point Veris at it with `--target your_module:answer`.

## How it works

```
golden_dataset/*.yaml  →  runner  →  DeepEval metrics
                                          ↓
                          SQLite trend store (git sha + timestamp)
                                          ↓
                          regression gate (floors + rolling band)
                                          ↓
                          report  →  exit 0 / 1
```

More in [ARCHITECTURE](docs/ARCHITECTURE.md) and [HARDENING](docs/HARDENING.md).

## CLI

| Command | What it does |
|---------|--------------|
| `veris run --target …` | Run the gate. Exit `0` pass, `1` fail |
| `veris run --dry-run` | Show planned cases and metrics, no LLM calls |
| `veris run --repeat N` | Flakiness audit (multiplies judge cost) |
| `veris run --csv-out path` | Also write a CSV report |
| `veris list-cases` | Cases, tags, and applicable blocking metrics |
| `veris validate` | Check dataset schema and duplicate IDs |
| `veris baseline` | Rolling median/stdev per case and metric |
| `veris trend` / `veris diff` | Run history and per-case score deltas |
| `veris usage` | Judge token and cost history |
| `veris cache-stats` / `cache-clear` | Manage the score cache |
| `veris prune --days N` | Trim old trend rows |

Makefile shortcuts exist too (`make run`, `make dry-run`, `make test`, `make seed`). Pass `TARGET=module:function` where needed.

## Configuration

Thresholds, weights, noise bands, and blocking vs advisory metrics live in [`veris/config.py`](veris/config.py). Nothing gate-critical is hardcoded in the runner.

| Env var | Purpose |
|---------|---------|
| `GROQ_API_KEY` | Judge credential |
| `OPENROUTER_API_KEY` | Judge credential. Set both keys for failover |
| `EVAL_JUDGE_PROVIDER` | `groq` (default) or `openrouter` |
| `EVAL_JUDGE_MODEL` | Groq model (default `llama-3.3-70b-versatile`) |
| `EVAL_OPENROUTER_MODEL` | OpenRouter model (default `openrouter/free`) |
| `GOLDEN_DATASET_DIR` | Cases directory (default `./golden_dataset`) |
| `TREND_DB_PATH` | SQLite path (default `./veris_trends.db`) |

## Golden dataset

YAML files in [`golden_dataset/`](golden_dataset/): `retrieval`, `reasoning`, `safety`, `tone`, `edge_cases`. Drop in more `*.yaml` or `*.yml` files and the loader picks them up.

```yaml
cases:
  - id: retrieval-001
    input: "What is the refund window for annual plans?"
    expected_output: "Annual plans can be refunded within 30 days of purchase."
    retrieval_context:
      - "Our refund policy: annual subscriptions are refundable within 30 days…"
    tags: [retrieval, billing]
```

Run a subset by tag:

```bash
veris run --target your_module:your_callable --tags retrieval,edge-case
```

**Metric applicability:** `faithfulness` and `hallucination` need `retrieval_context`. Without it they're N/A, not failures. Context-free cases (safety, tone, clarification) are gated on `answer_relevancy` and `toxicity` only.

## What blocks a run

A metric blocks when either:

1. **Floor:** the score is below `MetricSpec.min_score`, or
2. **Regression:** the score is more than `regression_stdev_band` standard deviations under that case's rolling median from the last `baseline_window` green runs (only once `min_baseline_samples` exist).

Advisory metrics are recorded and shown but never change the exit code.

## Reports

Rich terminal output, plus JSON, HTML (with a recent trend), and CSV. Reports are artifacts. The exit code and the SQLite run record are the source of truth.

## CI

This repo has no GitHub Actions workflow on purpose. Veris is a local, manual gate. Run it yourself:

```bash
make test
make TARGET=your_module:your_callable dry-run
make TARGET=your_module:your_callable run
```

If you embed Veris in a project, wrap the CLI in your own CI. The `0`/`1` exit code is built for that.

## Docker

```bash
docker build -t veris .
docker run --rm -e GROQ_API_KEY -v veris-data:/app/data veris \
  run --target your_module:your_callable
```

## Docs

- [User guide](docs/USER_GUIDE.md): install, dataset format, metrics, baselines, exit codes
- [Architecture](docs/ARCHITECTURE.md): pipeline, layout, gate semantics, persistence
- [Hardening](docs/HARDENING.md): logic-hole review and mitigations
- [Contributing](docs/CONTRIBUTING.md): setup, checks, adding cases and metrics

## Development

```bash
pip install -e ".[dev]"
make test
make lint
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Mass failures or empty scores | Missing or invalid API key | Set `GROQ_API_KEY`. The circuit breaker aborts after repeated judge failures |
| Daily quota hit mid-run | Primary provider out of tokens | Set the other provider's key. Veris falls back on daily limits, not short rate limits |
| OpenRouter `402` | Balance can't reserve `max_tokens` | Keep judge `max_tokens` low (built-in 256) or add credits |
| `invalid score` | Judge returned NaN/None/Inf | The case fails. Check the provider or clear the cache |
| Target ERROR `expected str` | Callable returned a non-string | Return a string |
| "metric missing" | Metric should apply but the suite errored | Check logs, try `--no-cache` |
| Safety/tone cases fail on hallucination | Old config without applicability rules | Upgrade |
| No regression sensitivity | Fewer than `min_baseline_samples` green runs | `make seed` |
| Stale scores after config change | Hot cache | `veris cache-clear` |

## License

MIT. See [LICENSE](LICENSE).
