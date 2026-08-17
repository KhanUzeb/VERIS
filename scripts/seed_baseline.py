"""
Run this once against a known-good build to seed Veris' trend store
baseline. Without at least `min_baseline_samples` green runs, the
regression gate falls back to absolute floors only (still safe, just
less sensitive).

Usage:
    python scripts/seed_baseline.py --target your_agent_module:your_callable --n 3
"""
import argparse
import importlib
import sys
from pathlib import Path

# Ensure project cwd is importable (same issue as the `veris` console script).
_cwd = str(Path.cwd().resolve())
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

from veris.report import print_report  # noqa: E402
from veris.runner import RunAborted, run_gate  # noqa: E402


def resolve(dotted):
    mod, _, fn = dotted.partition(":")
    if not mod or not fn:
        raise SystemExit("target must be 'module.path:function_name'")
    try:
        target = getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as e:
        raise SystemExit(f"could not load target {dotted!r}: {e}") from e
    if not callable(target):
        raise SystemExit(f"target {dotted!r} is not callable")
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Seed Veris trend baselines from green runs")
    p.add_argument("--target", required=True)
    p.add_argument("--n", type=int, default=3, help="number of seed runs (reduces noise)")
    args = p.parse_args()

    target_fn = resolve(args.target)
    for i in range(args.n):
        print(f"\n=== Veris seed run {i + 1}/{args.n} ===")
        try:
            report = run_gate(target_fn)
        except RunAborted as e:
            print_report(e.report)
            print(
                f"seed run {i + 1} aborted — {e}. Refusing to treat "
                "a partial run as baseline.",
                file=sys.stderr,
            )
            sys.exit(1)
        print_report(report)
        if not report.passed:
            print(
                f"seed run {i + 1} failed the gate — refusing to treat "
                "failed runs as baseline. Fix the target/scoring and retry.",
                file=sys.stderr,
            )
            sys.exit(1)
