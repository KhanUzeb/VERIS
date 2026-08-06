"""
Run this once against a known-good build to seed Veris' trend store
baseline. Without at least `min_baseline_samples` green runs, the
regression gate falls back to absolute floors only (still safe, just
less sensitive).

Usage:
    python scripts/seed_baseline.py --target veris.targets:my_agent_fn --n 3
"""
import argparse
import importlib
import sys

sys.path.insert(0, ".")

from veris.runner import run_gate  # noqa: E402
from veris.report import print_report  # noqa: E402


def resolve(dotted):
    mod, _, fn = dotted.partition(":")
    return getattr(importlib.import_module(mod), fn)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Seed Veris trend baselines from green runs")
    p.add_argument("--target", required=True)
    p.add_argument("--n", type=int, default=3, help="number of seed runs (reduces noise)")
    args = p.parse_args()

    target_fn = resolve(args.target)
    for i in range(args.n):
        print(f"\n=== Veris seed run {i + 1}/{args.n} ===")
        report = run_gate(target_fn)
        print_report(report)
        if not report.passed:
            print(
                f"seed run {i + 1} failed the gate — refusing to treat "
                "failed runs as baseline. Fix the target/scoring and retry.",
                file=sys.stderr,
            )
            sys.exit(1)
