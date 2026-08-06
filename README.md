# Veris

**Veris** is a local regression checker for LLM and agent systems.

It runs a golden dataset against your target, scores outputs with DeepEval and
RAGAS, compares results to a rolling SQLite baseline, and exits non-zero when
quality regresses. You run it when you want — nothing fires on commit or push.

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
        ├──► metrics.ragas     (context precision/recall, answer correctness)
        │
        ▼
veris.trend_store              # SQLite history keyed by git sha + timestamp
        │
        ▼
veris.regression_gate          # absolute floors + rolling median/stdev band
        │
        ▼
veris.langfuse_tracker         # optional traces + per-metric scores
        │
        ▼
veris.report                   # Rich terminal + JSON / HTML / CSV
        │
        ▼
   exit 0 / 1  ──► pass / fail (local or optional CI)
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

```bash
pip install -e .
# or: pip install -r requirements.txt

cp .env.example .env
# set GROQ_API_KEY (required); OPENROUTER_API_KEY optional fallback; LANGFUSE_* optional

# seed a baseline once against a known-good build
python scripts/seed_baseline.py --target veris.targets:my_agent_fn --n 3

# preview what will run (zero LLM calls)
veris run --target veris.targets:my_agent_fn --dry-run
# equivalent: python -m veris.cli run --target veris.targets:my_agent_fn --dry-run

# run the gate
veris run --target veris.targets:my_agent_fn
```

Wire your real system by replacing `veris.targets.my_agent_fn` with any
`Callable[[str], str]` (LangGraph, HTTP client, etc.).

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
| `LANGFUSE_PUBLIC_KEY` / `SECRET_KEY` / `HOST` | Optional tracing |

See [`.env.example`](.env.example).

## Metric applicability

Not every metric applies to every case. Metrics that need retrieval context
(`faithfulness`, `hallucination`, `context_*`, `answer_correctness`) are
**N/A** — not blocking failures — when `retrieval_context` is empty.

Context-free cases (safety / tone / clarification) are gated on
`answer_relevancy` and `toxicity` only.

## Golden dataset

See [`golden_dataset/sample_cases.yaml`](golden_dataset/sample_cases.yaml).
Add more `*.yaml` / `*.yml` files; the loader globs the directory. Each case
supports tags for subset runs:

```bash
veris run --target veris.targets:my_agent_fn --tags retrieval,edge-case
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

## Optional CI (manual only)

CI is **off by default**. Commits and pushes do not trigger anything.

If you want GitHub Actions later, [`.github/workflows/veris-gate.yml`](.github/workflows/veris-gate.yml)
is wired for **manual** runs only (`workflow_dispatch`):

1. Open **Actions → veris-gate → Run workflow**
2. Unit tests always run; the live gate runs only when `GROQ_API_KEY` is set
3. Reports upload as artifacts; trend DBs are cached across manual runs

Day to day, prefer local commands: `make test`, `make run`, `make seed`.

## Docker

```bash
docker build -t veris .
docker run --rm -e GROQ_API_KEY -v veris-data:/app/data veris
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
| Langfuse silent | Keys unset or SDK mismatch | Optional; Veris continues without tracing |

## License

See repository license file if present; otherwise all rights reserved by the authors.
