# Veris

**Veris** is an agent-agnostic, local LLM evaluation and regression testing
gate. It runs golden datasets against your callable, scores outputs with
DeepEval, compares them with rolling SQLite baselines, and returns reliable
CI-ready pass/fail exit codes.

It is deliberately small and explicit: no bundled agent, no hosted dashboard,
no automatic commit hooks, and no required observability service.

Veris is a **gate**, not a dashboard. It answers three questions:

1. Did quality regress vs the last known-good baseline?
2. Is this a real regression or judge-model noise?
3. What is the trend over recent runs?

## Architecture

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

More detail: [the architecture](docs/ARCHITECTURE.md) and the
[complete user guide](docs/USER_GUIDE.md).

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
| `GROQ_API_KEY` | Primary judge when `EVAL_JUDGE_PROVIDER=groq` |
| `OPENROUTER_API_KEY` | Fallback (or primary if provider=`openrouter`) |
| `EVAL_JUDGE_MODEL` | Model id (default `llama-3.3-70b-versatile`) |
| `EVAL_JUDGE_PROVIDER` | `groq` (default) or `openrouter` |
| `GOLDEN_DATASET_DIR` | YAML cases directory (default `./golden_dataset`) |
| `TREND_DB_PATH` | SQLite trend store (default `./veris_trends.db`) |

See [`.env.example`](.env.example).

## Metric applicability

Not every metric applies to every case. Metrics that need retrieval context
(`faithfulness`, `hallucination`) are
**N/A** — not blocking failures — when `retrieval_context` is empty.

Context-free cases (safety / tone / clarification) are gated on
`answer_relevancy` and `toxicity` only.

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

## What “blocking” means

A metric blocks the run when either:

1. **Absolute floor** — score &lt; `MetricSpec.min_score`, or
2. **Regression** — score more than `regression_stdev_band` standard deviations
   below that case’s rolling median from the last `baseline_window` **green**
   runs (only after `min_baseline_samples` exist).

Advisory metrics are recorded and shown but do not flip the exit code.

## Continuous integration

This repository intentionally has no GitHub Actions workflow. Veris is a
local/manual gate, and it does not run on commits or pushes. Run checks locally:

```powershell
make test
make TARGET=your_agent_module:your_callable dry-run
make TARGET=your_agent_module:your_callable run
```

Projects embedding Veris may add their own CI workflow around the CLI.

## Docker

```bash
docker build -t veris .
docker run --rm -e GROQ_API_KEY -v veris-data:/app/data veris \
  run --target your_agent_module:your_callable
```

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
| Mass failures / empty scores | Missing or invalid API key | Set `GROQ_API_KEY`; circuit breaker aborts after consecutive judge failures |
| Cases fail with “metric missing” | Metric should apply but suite errored | Check warnings in the log; try `--no-cache` |
| Safety/tone cases fail on hallucination | Old config without applicability | Upgrade; those metrics are N/A without context |
| No regression sensitivity | Fewer than `min_baseline_samples` green runs | `make seed` |
| Stale scores after config change | Cache still hot | `veris cache-clear` (config fingerprint also invalidates automatically) |

## License

See repository license file if present; otherwise all rights reserved by the authors.
