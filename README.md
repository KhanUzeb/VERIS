# Veris

> Local, agent-agnostic LLM evaluation and regression testing gate for AI
> agents, RAG pipelines, and callables.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-2.1.1-blue.svg)
[![Docs](https://img.shields.io/badge/docs-user%20guide-informational)](docs/USER_GUIDE.md)

**Veris** is an agent-agnostic, local LLM evaluation and regression testing
gate. It runs golden datasets against any callable, scores outputs with
DeepEval, compares them with rolling SQLite baselines, and returns reliable
CI-ready pass/fail exit codes.

It is deliberately small and explicit: no bundled agent, no hosted dashboard,
no automatic commit hooks, and no required observability service.

Veris is a **gate**, not a dashboard. It answers three questions:

1. Did quality regress vs the last known-good baseline?
2. Is this a real regression or judge-model noise?
3. What is the trend over recent runs?

## Features

- **Agent-agnostic** — bring any `Callable[[str], str]` (LangGraph, FastAPI
  clients, local agents, RAG pipelines, plain functions).
- **LLM-as-judge scoring** — DeepEval metrics: faithfulness, answer relevancy,
  hallucination, toxicity.
- **Regression detection** — absolute floors plus a rolling median / standard
  deviation band over recent **green** runs.
- **Noise-aware** — judge-model noise is separated from real regressions via
  configurable bands and optional flakiness audits (`--repeat N`).
- **Local-first** — SQLite trend store, content-addressed score cache, no cloud
  dependency. CI-ready exit codes (`0` / `1` / `2`).
- **Free-tier friendly** — Groq + OpenRouter judges with automatic failover
  and a circuit breaker that stops a broken run fast instead of burning budget.

## Table of contents

- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [CLI reference](#cli-reference)
- [Your target contract](#your-target-contract)
- [Configuration](#configuration)
- [Golden dataset](#golden-dataset)
- [What "blocking" means](#what-blocking-means)
- [Reports](#reports)
- [Continuous integration](#continuous-integration)
- [Docker](#docker)
- [Documentation](#documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Quickstart

```bash
pip install -e .
# or: pip install -r requirements.txt

cp .env.example .env
# set GROQ_API_KEY (required); OPENROUTER_API_KEY optional fallback

# seed a baseline once against a known-good build
python scripts/seed_baseline.py --target your_agent_module:your_callable --n 3

# preview what will run (zero LLM calls)
veris run --target your_agent_module:your_callable --dry-run
# equivalent: python -m veris.cli run --target your_agent_module:your_callable --dry-run

# run the gate; your callable lives in your own project/module
veris run --target your_agent_module:your_callable
```

Your target must be a callable accepting the case input string and returning
the target output string. Veris does not ship an agent implementation.

## How it works

```
golden_dataset/*.yaml          # cases: input, expected_output, context, tags
        │
        ▼
veris.runner                   # invoke target, collect actual_output
        │
        ├──► metrics.deepeval  (faithfulness, relevancy, hallucination, toxicity)
        │
        ▼
veris.trend_store              # SQLite history keyed by git sha + timestamp
        │
        ▼
veris.regression_gate          # absolute floors + rolling median/stdev band
veris.report                   # Rich terminal + JSON / HTML / CSV
        │
        ▼
   exit 0 / 1  ──► pass / fail (local)
```

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/HARDENING.md](docs/HARDENING.md), and
[docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## CLI reference

| Command | Purpose |
|---------|---------|
| `veris run --target …` | Full gate; exits `0` pass / `1` fail |
| `veris run --dry-run` | Plan cases + applicable metrics; no LLM calls |
| `veris run --repeat N` | Flakiness audit (multiplies judge cost) |
| `veris run --csv-out path` | Also write a flat CSV report |
| `veris list-cases` | Cases, tags, applicable blocking metrics |
| `veris baseline` | Rolling median/stdev per case+metric |
| `veris validate` | Schema / duplicate-id check (no LLM calls) |
| `veris trend` / `veris diff` | History and case-by-case score deltas |
| `veris usage` | Judge token / cost history |
| `veris cache-stats` / `cache-clear` | Score-cache management |
| `veris prune --days N` | Trim old trend-store rows |

Makefile shortcuts: `make run`, `make dry-run`, `make test`, `make seed`, …
For the target-dependent shortcuts, provide `TARGET=module:function`.

## Your target contract

Veris does not implement or call a specific agent framework. Your target is
just a callable that accepts one string and returns one string:

```python
def answer(case_input: str) -> str:
    return my_agent.invoke(case_input)
```

Run it with `--target your_module:answer`. This keeps Veris compatible with
LangGraph, FastAPI clients, local agents, RAG pipelines, and ordinary Python
functions without coupling the harness to any of them.

## Configuration

Thresholds and metric policy live in [`veris/config.py`](veris/config.py).
Nothing gate-critical is hardcoded in the runner — weights, noise bands, and
blocking vs advisory metrics are config-driven.

| Env var | Purpose |
|---------|---------|
| `GROQ_API_KEY` | Judge credential (primary or fallback) |
| `OPENROUTER_API_KEY` | Judge credential (primary or fallback); set both keys for automatic failover |
| `EVAL_JUDGE_MODEL` | Groq model id (default `llama-3.3-70b-versatile`) |
| `EVAL_OPENROUTER_MODEL` | OpenRouter model id (default `openrouter/free` free router) |
| `EVAL_JUDGE_PROVIDER` | `groq` (default) or `openrouter` |
| `GOLDEN_DATASET_DIR` | YAML cases directory (default `./golden_dataset`) |
| `TREND_DB_PATH` | SQLite trend store (default `./veris_trends.db`) |

See [`.env.example`](.env.example).

## Golden dataset

Cases live under [`golden_dataset/`](golden_dataset/), split by focus:

| File | Focus |
|------|--------|
| `retrieval.yaml` | Grounded QA / multi-hop |
| `reasoning.yaml` | Policy & plan reasoning |
| `safety.yaml` | Refusals / jailbreaks |
| `tone.yaml` | Empathy / register |
| `edge_cases.yaml` | Hallucination, negation, ambiguity |

Add more `*.yaml` / `*.yml` files; the loader globs the directory. Each case
supports tags for subset runs:

```bash
veris run --target your_agent_module:your_callable --tags retrieval,edge-case
```

```yaml
cases:
  - id: retrieval-001
    input: "What is the refund window for annual plans?"
    expected_output: "Annual plans can be refunded within 30 days of purchase."
    retrieval_context:
      - "Our refund policy: annual subscriptions are refundable within 30 days…"
    tags: [retrieval, billing]
```

### Metric applicability

Not every metric applies to every case. Metrics that need retrieval context
(`faithfulness`, `hallucination`) are **N/A** — not blocking failures — when
`retrieval_context` is empty.

Context-free cases (safety / tone / clarification) are gated on
`answer_relevancy` and `toxicity` only.

## What "blocking" means

A metric blocks the run when either:

1. **Absolute floor** — score &lt; `MetricSpec.min_score`, or
2. **Regression** — score more than `regression_stdev_band` standard deviations
   below that case's rolling median from the last `baseline_window` **green**
   runs (only after `min_baseline_samples` exist).

Advisory metrics are recorded and shown but do not flip the exit code.

## Reports

- **Terminal (Rich)** — case status, score summary, latency, blocking details.
- **JSON** — machine-readable full `RunReport`.
- **HTML** — shareable report with recent composite trend.
- **CSV** — one row per case/metric for spreadsheets and further analysis.

Reports are artifacts, not the source of truth. The exit code and SQLite run
record are authoritative.

## Continuous integration

This repository intentionally has no GitHub Actions workflow. Veris is a
local/manual gate, and it does not run on commits or pushes. Run checks locally:

```powershell
make test
make TARGET=your_agent_module:your_callable dry-run
make TARGET=your_agent_module:your_callable run
```

Projects embedding Veris may add their own CI workflow around the CLI — the
stable `0` / `1` exit code is designed for exactly that.

## Docker

```bash
docker build -t veris .
docker run --rm -e GROQ_API_KEY -v veris-data:/app/data veris \
  run --target your_agent_module:your_callable
```

## Documentation

| Document | What it covers |
|----------|----------------|
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Install, target contract, dataset format, metrics, baselines, exit codes, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, package layout, gate semantics, resilience, persistence |
| [docs/HARDENING.md](docs/HARDENING.md) | Logic-hole review notes and mitigations |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Setup, checks, guidelines, adding cases/metrics, releases |

## Development

```bash
pip install -e ".[dev]"
make test          # or: pytest -q
make lint
```

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Mass failures / empty scores | Missing or invalid API key | Set `GROQ_API_KEY` (and optionally `OPENROUTER_API_KEY` for failover); circuit breaker aborts after consecutive judge failures |
| Daily quota / TPD mid-run | Primary provider out of daily tokens | Set the other provider's key; Veris falls back on TPD (not short TPM) |
| OpenRouter `402` / "can only afford" | Credit balance cannot reserve `max_tokens` | Keep judge `max_tokens` low (built-in 256) or add OpenRouter credits |
| `invalid score` in case detail | Judge returned NaN/None/Inf | Fail the case (not green-pass); check provider / clear cache |
| Target ERROR `expected str` | Callable returned a non-string | Return a string from your target |
| Cases fail with "metric missing" | Metric should apply but suite errored | Check warnings in the log; try `--no-cache` |
| Safety/tone cases fail on hallucination | Old config without applicability | Upgrade; those metrics are N/A without context |
| No regression sensitivity | Fewer than `min_baseline_samples` green runs | `make seed` |
| Stale scores after config change | Cache still hot | `veris cache-clear` (config fingerprint also invalidates automatically) |

## License

MIT — see [LICENSE](LICENSE).
