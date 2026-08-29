# Veris Architecture

Veris is a **local regression gate** for LLM and agent evaluation: golden
cases → target execution → LLM-as-judge scoring → baseline comparison →
pass/fail exit code.

For installation, target integration, dataset authoring, commands, and
troubleshooting, see [the user guide](USER_GUIDE.md). For an overview, start
at the [README](../README.md).

## Pipeline

```
┌─────────────────┐
│  Golden YAML    │  id, input, expected_output?, retrieval_context?, tags
└────────┬────────┘
         │ load + tag filter + unique-id check
         ▼
┌─────────────────┐
│  TargetFn       │  Callable[[str], str]: your agent / RAG / API
└────────┬────────┘
         │ ThreadPoolExecutor (max_workers)
         ▼
┌─────────────────┐
│  Score cache    │  SHA256(case + output + judge + metrics fingerprint)
└────────┬────────┘
         │ miss → DeepEval (shared judge, concurrency semaphore)
         ▼
┌─────────────────┐
│  Regression     │  absolute min_score + rolling median ± stdev band
│  gate           │  blocking vs advisory; N/A metrics skipped
└────────┬────────┘
         │ incremental SQLite writes (crash-safe)
         ▼
┌─────────────────┐
│  Reports        │  Rich / JSON / HTML / CSV
└─────────────────┘
```

## Package layout

| Module | Role |
|--------|------|
| `veris.cli` | Click CLI (`veris` / `python -m veris.cli`) |
| `veris.runner` | Orchestration, circuit breaker, flakiness repeats |
| `veris.regression_gate` | Pass/fail + composite score |
| `veris.trend_store` | SQLite baselines, history, usage, prune |
| `veris.cache` | Content-addressable score cache |
| `veris.judge_model` | Groq / OpenRouter judge + token accounting |
| `veris.metrics.*` | DeepEval suite |
| `veris.report` | Terminal / JSON / HTML / CSV |
| `veris.golden_dataset` | YAML loader |
| `veris.config` | All thresholds (`MetricSpec`, `Settings`) |
| `veris.flakiness` | `--repeat N` spread analysis |
| user target | Callable supplied via `--target module:function` |

## Gate semantics

Two independent checks per applicable `(case, metric)`:

1. **Absolute floor**: `score < MetricSpec.min_score` always fails (only blocking metrics flip the exit code).
2. **Regression**: once enough green history exists, fail when the score falls more than `band * stdev` below the case's rolling median. Flat baselines (`stdev == 0`) use a synthetic relative stdev so `stdev_band_override` still applies; at the default band the floor is `0.9 * median`.

Metrics with `requires_retrieval_context` / `requires_expected_output` that the case cannot satisfy are marked **N/A** and never treated as missing blocking scores.

## Resilience

- A circuit breaker aborts the run after consecutive judge failures (bad key or outage) instead of reporting a mass false regression. A cooperative cancel stops further judge work, and the CLI still writes JSON/HTML for the partial run.
- The score cache skips judge calls when the inputs and metric config fingerprint are unchanged, and it refuses non-finite cached values.
- Incremental DB writes keep completed cases safe through mid-run crashes. `(run, case, metric)` rows upsert under a unique index.
- Trend-store rows always use UTC ISO-8601 timestamps so `ORDER BY` and prune stay consistent.
- Provider fallback covers Groq (`EVAL_JUDGE_MODEL`) and OpenRouter (`EVAL_OPENROUTER_MODEL`, default `openrouter/free`). The primary is set by `EVAL_JUDGE_PROVIDER`; the other is used on daily quota or hard failures when its key is set. Short TPM retries stay on the primary (see [USER_GUIDE](USER_GUIDE.md#judge-providers-groq--openrouter)).
- Non-finite scores (`NaN`, `Inf`, `None`) from the judge fail the gate instead of silently passing floor checks.

Hardening history and rationale: [HARDENING.md](HARDENING.md).

## Persistence

- Trend DB: `TREND_DB_PATH` (default `./veris_trends.db`)
- Cache DB: sibling `*_cache.db`
- Reports: `veris_report.json` / `.html` (+ optional `.csv`)
