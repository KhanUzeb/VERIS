# Hardening notes

Review pass against logic holes that could make the gate green-pass,
corrupt baselines, or fail opaquely. Fixes landed in code; this doc is
the durable record of *what* broke and *why* the change exists.

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
| Corrupt NaN entries in score cache | Replayed into the gate | Cache `get`/`set` refuse non-finite values |
| `VERIS_MAX_RUN_COST_USD` garbage | Import-time `float()` crash | Validated in `Settings.__post_init__` with a clear error |
| `--tags ,,,` / empty tags | Filtered to nothing / surprising empty runs | Reject empty tag lists |
| HTML report ignored `cr.error` | Showed `FAIL` instead of `ERROR` | Shared `_case_status()` for terminal/CSV/HTML |
| Seed script bad `--target` | Raw `AttributeError` | Same validation as CLI |

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
