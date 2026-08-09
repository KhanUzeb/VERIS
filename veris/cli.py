from __future__ import annotations

import importlib
import subprocess
import sys

import click
from rich.console import Console
from rich.table import Table

from .cache import ScoreCache
from .config import SETTINGS
from .judge_model import usage_totals
from .langfuse_tracker import LangfuseTracker
from .report import print_report, write_csv_report, write_html_report, write_json_report
from .runner import run_gate
from .trend_store import TrendStore

console = Console()


def _resolve_target(dotted_path: str):
    """'module.path:function_name' -> the actual callable."""
    module_path, _, func_name = dotted_path.partition(":")
    if not module_path or not func_name:
        raise click.BadParameter(
            "target must be 'module.path:function_name'", param_hint="--target"
        )
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return None


@click.group()
def cli():
    """Veris - local regression gate for LLM/agent systems (DeepEval + RAGAS + Langfuse)."""


@cli.command()
@click.option("--target", required=True, help="dotted path: module.path:function_name")
@click.option("--tags", default=None, help="comma-separated tags to filter golden cases")
@click.option("--json-out", default="veris_report.json", help="path for JSON report")
@click.option("--html-out", default="veris_report.html", help="path for HTML report")
@click.option("--csv-out", default=None, help="optional path for CSV report (one row per case/metric)")
@click.option("--no-langfuse", is_flag=True, help="skip pushing traces to Langfuse")
@click.option("--no-cache", is_flag=True, help="bypass the score cache, force fresh judge calls")
@click.option("--repeat", default=1, help="score each case N times to detect judge-model flakiness "
                                           "(disables cache, multiplies cost — use selectively)")
@click.option("--fail-fast", is_flag=True, help="stop after the first blocking case failure")
@click.option("--dry-run", is_flag=True, help="load cases + resolve target, print plan, make zero LLM calls")
def run(target, tags, json_out, html_out, csv_out, no_langfuse, no_cache, repeat, fail_fast, dry_run):
    """Run the golden set against TARGET and gate on regressions."""
    target_fn = _resolve_target(target)
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    if dry_run:
        from .golden_dataset import load_golden_cases
        cases = load_golden_cases(SETTINGS.golden_dataset_dir, tags=tag_list)
        console.print(f"[bold]dry-run[/bold] target={target}  cases={len(cases)}  tags={tag_list or 'all'}")
        console.print(f"judge={SETTINGS.judge_provider}/{SETTINGS.judge_model}")
        console.print(f"dataset={SETTINGS.golden_dataset_dir}")
        table = Table(title="planned cases")
        table.add_column("id")
        table.add_column("tags")
        table.add_column("applicable metrics")
        for c in cases:
            applicable = [m.name for m in SETTINGS.metrics if m.is_applicable(c)]
            table.add_row(c.id, ",".join(c.tags) or "-", ", ".join(applicable))
        console.print(table)
        return

    report = run_gate(
        target_fn, tags=tag_list, git_sha=_git_sha(),
        use_cache=not no_cache, repeat=repeat, fail_fast=fail_fast,
    )

    print_report(report)
    write_json_report(report, json_out)
    hist_store = TrendStore()
    history = hist_store.recent_runs(limit=20)
    hist_store.close()
    write_html_report(report, html_out, trend_history=history)
    written = [json_out, html_out]
    if csv_out:
        write_csv_report(report, csv_out)
        written.append(csv_out)
    click.echo(f"\nreports written: {', '.join(written)}")

    totals = usage_totals()
    if totals["calls"]:
        click.echo(
            f"judge usage: {totals['calls']} calls, {totals['prompt_tokens']}+"
            f"{totals['completion_tokens']} tokens, ~${totals['estimated_cost_usd']}"
        )

    if not no_langfuse:
        LangfuseTracker().log_run(report)

    sys.exit(0 if report.passed else 1)


@cli.command("list-cases")
@click.option("--dataset-dir", default=None, help="defaults to SETTINGS.golden_dataset_dir")
@click.option("--tags", default=None, help="comma-separated tags to filter")
def list_cases(dataset_dir, tags):
    """List golden cases with tags and which metrics apply."""
    from .golden_dataset import load_golden_cases

    target_dir = dataset_dir or SETTINGS.golden_dataset_dir
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    cases = load_golden_cases(target_dir, tags=tag_list)

    table = Table(title=f"golden cases ({len(cases)})")
    table.add_column("id")
    table.add_column("tags")
    table.add_column("context")
    table.add_column("expected")
    table.add_column("applicable (blocking)")
    for c in cases:
        applicable = [
            m.name for m in SETTINGS.metrics if m.is_applicable(c) and m.blocking
        ]
        table.add_row(
            c.id,
            ",".join(c.tags) or "-",
            "yes" if c.retrieval_context else "no",
            "yes" if c.expected_output else "no",
            ", ".join(applicable) or "-",
        )
    console.print(table)


@cli.command()
@click.option("--case", "case_id", default=None, help="filter to one case id")
@click.option("--metric", default=None, help="filter to one metric name")
@click.option("--window", default=None, type=int, help="baseline window (default: config)")
def baseline(case_id, metric, window):
    """Show rolling baselines (median/stdev/n) from green runs."""
    store = TrendStore()
    window = window or SETTINGS.baseline_window
    from .golden_dataset import load_golden_cases

    cases = load_golden_cases(SETTINGS.golden_dataset_dir)
    if case_id:
        cases = [c for c in cases if c.id == case_id]
        if not cases:
            click.echo(f"no case with id {case_id!r}")
            store.close()
            return

    metrics = [m.name for m in SETTINGS.metrics]
    if metric:
        metrics = [metric]

    table = Table(title=f"baselines (window={window}, green runs only)")
    table.add_column("case")
    table.add_column("metric")
    table.add_column("median", justify="right")
    table.add_column("stdev", justify="right")
    table.add_column("n", justify="right")
    table.add_column("status")

    any_row = False
    for c in cases:
        for m in metrics:
            if not any(spec.name == m and spec.is_applicable(c) for spec in SETTINGS.metrics):
                # still show if user asked for a specific metric
                if metric is None:
                    continue
            bas = store.baseline_for_case_metric(c.id, m, window)
            if bas is None:
                table.add_row(c.id, m, "-", "-", "0", "no data")
            else:
                med, std, n = bas
                status = "ready" if n >= SETTINGS.min_baseline_samples else f"need {SETTINGS.min_baseline_samples - n} more"
                table.add_row(c.id, m, f"{med:.3f}", f"{std:.3f}", str(n), status)
            any_row = True

    store.close()
    if not any_row:
        click.echo("no baseline rows to show")
    else:
        console.print(table)


@cli.command()
@click.option("--limit", default=15)
def trend(limit):
    """Show recent run history from the trend store."""
    store = TrendStore()
    runs = store.recent_runs(limit=limit)
    store.close()
    if not runs:
        click.echo("no runs recorded yet")
        return
    for r in runs:
        status = "PASS" if r["passed"] else "FAIL"
        click.echo(
            f"{r['timestamp']:<26} {status:<5} composite={r['composite_score']:.3f} "
            f"failed={r['failed_cases']} regressed={r['regressed_cases']} sha={r['git_sha'] or 'n/a'}"
        )


@cli.command()
@click.argument("run_a", default="previous")
@click.argument("run_b", default="latest")
def diff(run_a, run_b):
    """Compare two runs case-by-case. Accepts run_id, git sha, or the
    keywords 'latest'/'previous'. Default: diff previous vs latest."""
    store = TrendStore()
    id_a = store.resolve_run_id(run_a)
    id_b = store.resolve_run_id(run_b)
    if not id_a or not id_b:
        click.echo(f"could not resolve one or both runs: {run_a!r}={id_a}, {run_b!r}={id_b}")
        store.close()
        return

    scores_a = store.get_case_scores(id_a)
    scores_b = store.get_case_scores(id_b)
    meta_a, meta_b = store.get_run(id_a), store.get_run(id_b)
    store.close()

    table = Table(title=f"diff {id_a} ({meta_a['timestamp']}) -> {id_b} ({meta_b['timestamp']})")
    table.add_column("case")
    table.add_column("metric")
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("delta", justify="right")

    all_case_ids = sorted(set(scores_a) | set(scores_b))
    any_change = False
    for case_id in all_case_ids:
        metrics = sorted(set(scores_a.get(case_id, {})) | set(scores_b.get(case_id, {})))
        for m in metrics:
            before = scores_a.get(case_id, {}).get(m)
            after = scores_b.get(case_id, {}).get(m)
            if before is None or after is None:
                delta_str = "new" if before is None else "removed"
                table.add_row(case_id, m, f"{before:.3f}" if before is not None else "-",
                              f"{after:.3f}" if after is not None else "-", delta_str)
                any_change = True
                continue
            delta = after - before
            if abs(delta) < 0.005:
                continue
            any_change = True
            color = "red" if delta < 0 else "green"
            table.add_row(case_id, m, f"{before:.3f}", f"{after:.3f}", f"[{color}]{delta:+.3f}[/{color}]")

    if not any_change:
        click.echo("no meaningful score changes between these runs")
    else:
        console.print(table)

    console.print(
        f"\ncomposite: {meta_a['composite_score']:.3f} -> {meta_b['composite_score']:.3f}   "
        f"({meta_b['composite_score'] - meta_a['composite_score']:+.3f})"
    )


@cli.command("cache-stats")
def cache_stats():
    """Show score cache hit stats (how much judge-model cost is being saved)."""
    cache = ScoreCache()
    stats = cache.stats()
    cache.close()
    click.echo(f"cached entries: {stats['cached_entries']}   total hits: {stats['total_hits']}")


@cli.command("cache-clear")
def cache_clear():
    """Clear the score cache — use after changing metric logic or judge model."""
    cache = ScoreCache()
    cache.clear()
    cache.close()
    click.echo("cache cleared")


@cli.command()
@click.option("--dataset-dir", default=None, help="defaults to SETTINGS.golden_dataset_dir")
def validate(dataset_dir):
    """Check the golden dataset for schema errors and duplicate ids.
    Makes zero LLM calls — safe while authoring new test cases
    without burning judge-model budget."""
    from .golden_dataset import load_golden_cases

    target_dir = dataset_dir or SETTINGS.golden_dataset_dir
    try:
        cases = load_golden_cases(target_dir)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[bold red]validation failed:[/bold red] {e}")
        sys.exit(1)

    tag_counts: dict[str, int] = {}
    context_free = 0
    for c in cases:
        for t in c.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        if not c.retrieval_context:
            context_free += 1

    console.print(f"[bold green]{len(cases)} valid cases[/bold green] in {target_dir}")
    console.print("tags: " + ", ".join(f"{k}={v}" for k, v in sorted(tag_counts.items())))
    if context_free:
        na_metrics = [
            m.name for m in SETTINGS.metrics if m.requires_retrieval_context
        ]
        console.print(
            f"[yellow]{context_free} case(s) have no retrieval_context — "
            f"these metrics are N/A (not blocking): {', '.join(na_metrics)}[/yellow]"
        )


@cli.command()
@click.option("--limit", default=15)
def usage(limit):
    """Show judge-model token usage and estimated cost per run."""
    store = TrendStore()
    rows = store.usage_history(limit=limit)
    store.close()
    if not rows:
        click.echo("no usage data recorded yet")
        return
    total_cost = sum(r["estimated_cost_usd"] for r in rows)
    for r in rows:
        click.echo(
            f"{r['timestamp']:<26} calls={r['calls']:<5} "
            f"tokens={r['prompt_tokens']}+{r['completion_tokens']:<8} "
            f"~${r['estimated_cost_usd']:<8.4f} sha={r['git_sha'] or 'n/a'}"
        )
    click.echo(f"\ntotal across last {len(rows)} runs: ~${total_cost:.4f}")


@cli.command()
@click.option("--days", default=90, help="delete runs older than this many days")
@click.option("--yes", is_flag=True, help="skip confirmation prompt")
def prune(days, yes):
    """Delete old runs from the trend store. The DB grows unbounded
    otherwise — this trims it while leaving recent history (and the
    regression baseline window, which only looks back baseline_window
    runs anyway) untouched."""
    if not yes and not click.confirm(f"delete all runs older than {days} days?"):
        return
    store = TrendStore()
    deleted = store.prune_older_than(days)
    store.close()
    click.echo(f"pruned {deleted} run(s) older than {days} days")


if __name__ == "__main__":
    cli()
