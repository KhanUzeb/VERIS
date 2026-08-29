# Veris Hardening Notes

Review pass against logic holes that could make the gate green-pass,
corrupt baselines, or fail opaquely. Fixes landed in code; this doc is
the durable record of *what* broke and *why* the change exists.

Part of the [Veris documentation](../README.md#documentation).

## Gate holes

| Issue | Risk | Fix |
|-------|------|-----|
| `NaN` / `None` / `Inf` metric scores | Python `NaN < threshold` is always `False`, so a broken judge could green-pass absolute floors | `evaluate_case` strips non-finite values, treats them as missing/blocking; DeepEval wrapper rejects non-numeric scores before return |
| Empty score dict | Already failed the gate; kept | unchanged |
| Missing applicable blocking metrics | Already failed the gate; kept | unchanged |

## Persistence / baseline integrity

| Issue | Risk | Fix |
|-------|------|-----|
| Mixed timestamps (`datetime('now')` vs `isoformat()`) | Lexicographic `ORDER BY timestamp` and prune cutoffs were inconsistent across stub vs finalized runs | All writes use UTC ISO-8601 via `_utc_now_iso()` / aware `isoformat()` |
| Duplicate `(run_id, case_id, metric_name)` rows | Baseline `n` inflated; `diff` last-write-wins silently | Unique index + `ON CONFLICT` upsert; legacy duplicates collapsed on open |
| Foreign keys off (SQLite default) | Orphan `case_scores` with no parent `runs` row | `PRAGMA foreign_keys=ON` |
| `veris prune --days 0` | `timestamp < now` wiped the entire history | CLI min `1`; `prune_older_than` raises if `days < 1` |

## Runner / judge / CLI

| Issue | Risk | Fix |
|-------|------|-----|
| Target returned non-`str` | Opaque DeepEval / cache failures | Case marked `ERROR` with a clear message |
| Judge empty / null `message.content` | Downstream metric crashes | Treated as provider failure (retry / circuit breaker) |
| Groq TPD (daily) quota + short backoff | Run hung retrying for ~20s then aborted; fallback never used | Parse provider "try again in …" for TPM; on tokens-per-day exhaustion skip primary retries and use configured OpenRouter/Groq fallback |
| Corrupt NaN entries in score cache | Replayed into the gate | Cache `get`/`set` refuse non-finite values |
| `VERIS_MAX_RUN_COST_USD` garbage | Import-time `float()` crash | Validated by `_parse_max_run_cost_usd()` at import (not `__post_init__`, so `dataclasses.replace` in tests is not overwritten) |
| `--tags ,,,` / empty tags | Filtered to nothing / surprising empty runs | Reject empty tag lists |
| HTML report ignored `cr.error` | Showed `FAIL` instead of `ERROR` | Shared `_case_status()` for terminal/CSV/HTML |
| Seed script bad `--target` | Raw `AttributeError` | Same validation as CLI |
| `dependencies` nested under `[project.urls]` | `pip install -e .` installed the package with **no** runtime deps | `dependencies` lives under `[project]` |
| One DeepEval metric raising | Entire case score dict discarded; circuit breaker could trip on a single flaky metric | Each metric is isolated; empty suite still counts as a judge failure |
| Circuit-breaker abort skipped CLI reports | SQLite had the partial run, but JSON/HTML were never written | `RunAborted` carries the report; CLI/seed still emit artifacts then fail |
| Stacked retry layers on the judge | `@retry` decorator on `generate()` combined with the inner `_call_primary_with_backoff` loop → up to ~15 primary calls + 2 fallbacks per case/metric; made judge spend and attempt count unbounded and unpredictable (undermines `VERIS_MAX_RUN_COST_USD`) | Removed the tenacity decorator; all primary retry/backoff is now owned solely by `_call_primary_with_backoff` (5 attempts, exponential, honors "try again in …") — fallback still fires once on daily-quota / connection failure |

## What was already solid

- Empty scores fail the gate
- Metric applicability (N/A without retrieval context)
- Circuit breaker + fail-fast + cost ceiling abort paths persist partial runs
- Score cache requires the full applicable metric set before reuse
- Flat baselines still honor `stdev_band_override` via synthetic relative stdev

## Verification

```powershell
make test
# or: pytest -q
```
