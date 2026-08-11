# Veris User Guide

Veris is a local, agent-agnostic regression gate for LLM and agent systems.
It is the test harness around your system, not the system itself.

You provide:

- a golden YAML dataset;
- a Python callable that sends one case input to your agent and returns its output;
- credentials for the LLM judge.

Veris provides the repeatable execution, DeepEval scoring, cache, SQLite
baseline, regression decision, reports, and process exit code.

## What happens during a run

```text
golden YAML
    ↓
load + validate + optional tag filter
    ↓
your target: callable[str, str]
    ↓
DeepEval judge metrics
    ↓
absolute floors + rolling green baseline comparison
    ↓
SQLite history + JSON/HTML/CSV reports
    ↓
exit 0 for pass, exit 1 for regression/failure
```

Each case runs independently. Cases are executed concurrently, judge calls
are concurrency-limited, completed case scores are written incrementally, and
a circuit breaker stops a run when the judge provider is clearly unavailable.

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On Unix-like systems, activate with `source .venv/bin/activate`.

Copy the configuration template:

```powershell
Copy-Item .env.example .env
```

### Judge providers (Groq + OpenRouter)

Set **both** API keys so the primary can fail over to the other.

| Setting | Role |
|---------|------|
| `EVAL_JUDGE_PROVIDER` | Primary: `groq` (default) or `openrouter` |
| `GROQ_API_KEY` | Groq credential |
| `OPENROUTER_API_KEY` | OpenRouter credential |
| `EVAL_JUDGE_MODEL` | Groq model id (default `llama-3.3-70b-versatile`) |
| `EVAL_OPENROUTER_MODEL` | OpenRouter model id (default `openrouter/free`) |

`openrouter/free` is OpenRouter’s [Free Models Router](https://openrouter.ai/docs/guides/routing/routers/free-router) — it auto-picks an available free model. Pin a free variant with a `:free` suffix if you prefer (e.g. `meta-llama/llama-3.3-70b-instruct:free`).

Fallback behavior when the other key is set:

- **Daily quota (TPD/RPD), network errors, empty judge replies** → try the other provider once.
- **Short TPM rate limits** → retry primary only (honors “try again in …”, capped). No fallback.
- Each provider keeps its own model id (Groq ≠ OpenRouter naming).
- Cross-provider failover is **opt-out** via `EVAL_JUDGE_FAILOVER=0` (default on). Failover sends case prompts to the other provider — only enable when that is acceptable.
- Judge `max_tokens` is capped at 256 so thin free-tier balances do not 402 on reserved completion tokens.
- Score-cache keys include the failover model id when failover is enabled, so primary-only and failover scores do not collide.

## The target contract

Veris does not ship an agent adapter. Your target is any importable Python
callable with this shape:

```python
def answer(case_input: str) -> str:
    """Call the system under test and return its final text output."""
    return my_agent.invoke(case_input)
```

If the callable lives in `my_eval_target.py`, run it as:

```powershell
veris run --target my_eval_target:answer
```

The callable may call LangGraph, an HTTP API, a RAG pipeline, a local model,
or ordinary application code. Veris only requires the string-in/string-out
boundary. Keep authentication, retries, and system-specific setup in your
target module.

## First checks

Before spending judge tokens:

```powershell
veris validate
veris list-cases
veris run --target my_eval_target:answer --dry-run
```

`validate` checks YAML shape and duplicate IDs. `list-cases` shows tags,
context availability, expected-output availability, and blocking metrics.
`dry-run` shows the execution plan and makes no target or judge calls.

## Golden dataset format

Files under `golden_dataset/` must use `.yaml` or `.yml`. A file can contain a
list directly or a mapping with a `cases` list:

```yaml
cases:
  - id: refund-window
    input: "What is the refund window for annual plans?"
    expected_output: "Annual plans can be refunded within 30 days."
    retrieval_context:
      - "Annual subscriptions are refundable within 30 days of purchase."
    tags: [retrieval, billing]
    notes: "Checks grounded policy retrieval."
```

Fields:

| Field | Required | Meaning |
|---|---:|---|
| `id` | yes | Globally unique case ID across all dataset files. |
| `input` | yes | String passed to your target callable. |
| `expected_output` | no | Reference answer used by the case’s applicable judge logic. |
| `retrieval_context` | no | Source passages available to grounded metrics. |
| `tags` | no | Labels used by `--tags`. |
| `notes` | no | Human documentation; not sent as the target input. |

Cases without `retrieval_context` are still valid. Faithfulness and
hallucination are marked N/A and do not block those cases. Context-free cases
are gated by answer relevancy and toxicity.

IDs must be unique across every YAML file. A malformed file, wrong top-level
shape, invalid case, or duplicate ID makes `validate` fail before any LLM call.

## Metrics and gate policy

The default DeepEval policy is defined in `veris/config.py`:

| Metric | Blocking | Default floor | Applies when |
|---|---:|---:|---|
| `faithfulness` | yes | 0.75 | Retrieval context exists. |
| `answer_relevancy` | yes | 0.70 | Every case. |
| `hallucination` | yes | 0.80 | Retrieval context exists; returned inverted so higher is better. |
| `toxicity` | yes | 0.90 | Every case; returned inverted so higher is better. |

The gate performs two checks for each applicable metric:

1. Absolute floor: a score below `min_score` fails a blocking metric.
2. Regression: after enough green history exists, a score below the rolling
   median minus the configured noise band fails.

Only green runs contribute to the baseline. A failed run cannot redefine
“normal.” Until `min_baseline_samples` green runs exist, the absolute floors
are the protection mechanism.

Tune policy only in `veris/config.py`. `metrics_fingerprint()` makes metric
configuration changes invalidate old score-cache entries automatically.

## Establishing a baseline

Run a known-good version of your target several times:

```powershell
python scripts/seed_baseline.py --target my_eval_target:answer --n 3
```

The script refuses to treat a failed run as a baseline. Inspect the baseline:

```powershell
veris baseline
veris baseline --case refund-window --metric faithfulness
```

Use a baseline from a stable, trusted build. Do not seed it from a degraded
or partially configured environment.

## Normal runs and exit codes

```powershell
veris run --target my_eval_target:answer
```

Exit codes:

- `0`: every case passed its blocking checks;
- `1`: a blocking floor/regression failed, a target failed, or the judge
  circuit breaker aborted the run;
- `2`: invalid CLI arguments.

Useful options:

```powershell
# Run only selected cases
veris run --target my_eval_target:answer --tags retrieval,billing

# Force fresh judge scores
veris run --target my_eval_target:answer --no-cache

# Stop after the first blocking case failure
veris run --target my_eval_target:answer --fail-fast

# Repeat scoring to inspect judge noise
veris run --target my_eval_target:answer --repeat 3

# Add a flat metric report
veris run --target my_eval_target:answer --csv-out run.csv
```

`--repeat` is an audit mode: it increases judge cost and does not use the
score cache. The mean score is used for that run and spread information is
reported.

## Cache and persistence

The score cache is content-addressed by case data, actual output, judge
identity, and metric configuration. It skips unchanged judge work.

```powershell
veris cache-stats
veris cache-clear
```

The trend store is SQLite at `TREND_DB_PATH`, defaulting to
`./veris_trends.db`. A sibling cache database is created beside it. Reports
default to `veris_report.json` and `veris_report.html`.

Inspect history and cost:

```powershell
veris trend
veris diff previous latest
veris usage
veris prune --days 90 --yes
```

## Reports

- Terminal report: case status, score summary, latency, and blocking details.
- JSON: machine-readable full `RunReport`.
- HTML: shareable report with recent composite trend.
- CSV: one row per case/metric for spreadsheets and further analysis.

Reports are artifacts, not the source of truth for the gate. The exit code
and SQLite run record are authoritative.

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | empty | Primary judge credential when provider is Groq. |
| `OPENROUTER_API_KEY` | empty | OpenRouter credential (use with Groq for failover). |
| `EVAL_JUDGE_PROVIDER` | `groq` | `groq` or `openrouter`. |
| `EVAL_JUDGE_MODEL` | `llama-3.3-70b-versatile` | Groq judge model ID. |
| `EVAL_OPENROUTER_MODEL` | `openrouter/free` | OpenRouter judge model / free router. |
| `GOLDEN_DATASET_DIR` | `./golden_dataset` | Dataset directory. |
| `TREND_DB_PATH` | `./veris_trends.db` | Trend SQLite path. |
| `VERIS_MAX_RUN_COST_USD` | unset | Optional estimated judge-cost ceiling. |

The remaining gate knobs—baseline window, minimum samples, concurrency,
circuit-breaker threshold, metric floors, weights, and noise bands—are
versioned in `veris/config.py` so changes are reviewable.

## Continuous integration

This repository intentionally ships no GitHub Actions workflow. Veris is a
local/manual gate; it does not run on commits or pushes. Consuming projects
can wrap the CLI in their own CI workflow and provide their own
`module:function` target there.

## Repository map

| Path | Responsibility |
|---|---|
| `veris/cli.py` | Click commands, options, exit codes, and command UX. |
| `veris/runner.py` | Target execution, concurrency, scoring orchestration, circuit breaker. |
| `veris/metrics/deepeval_metrics.py` | DeepEval metric construction and score normalization. |
| `veris/judge_model.py` | Groq/OpenRouter judge adapter, retries, fallback, usage totals. |
| `veris/regression_gate.py` | Floors, rolling-baseline comparisons, composite score. |
| `veris/trend_store.py` | SQLite runs, case scores, baselines, usage, and pruning. |
| `veris/cache.py` | Content-addressed score cache. |
| `veris/golden_dataset.py` | YAML loading, Pydantic validation, duplicate detection. |
| `veris/schemas.py` | Dataset, case-result, and run-report models. |
| `veris/report.py` | Terminal, JSON, HTML, and CSV output. |
| `veris/flakiness.py` | Repeat-score spread analysis. |
| `docs/HARDENING.md` | Logic-hole review notes and mitigations. |
| `golden_dataset/` | Golden cases supplied to the harness. |
| `scripts/seed_baseline.py` | Known-good baseline seeding helper. |
| `tests/` | Offline regression tests for the harness itself. |

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `could not load target` | Wrong import path or non-callable symbol. | Check `module:function`; run from the project environment. |
| Empty scores / circuit breaker | Judge key, model, provider, or network failure. | Test provider access and credentials; do not accept an empty-score run. |
| `invalid score` / NaN in detail | Judge returned non-finite metric values. | Treat as judge failure; check provider responses / retries. |
| Target `expected str` ERROR | Callable returned a non-string. | Coerce to `str` in your target wrapper. |
| Accidental full history wipe | `--days 0` used to be allowed. | Prune requires `--days >= 1`. |
| Many metric-missing messages | Judge suite failed partially or cache contains old data. | Run `veris cache-clear`, then retry. |
| No regression sensitivity | Not enough green baseline samples. | Seed at least `min_baseline_samples` trusted runs. |
| Target output is stale | Target or wrapper is caching its own response. | Disable target-side caching for evaluation or include version inputs. |
| Run costs too much | Too many cases, repeats, or workers. | Use tags, avoid `--repeat`, and set `VERIS_MAX_RUN_COST_USD`. |
