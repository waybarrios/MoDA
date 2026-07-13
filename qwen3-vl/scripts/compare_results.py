"""Compare lmms-eval results between Base and MoDA models.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --base eval_results/base --moda eval_results/moda
"""

import argparse
import json
from pathlib import Path


# Expected metrics per benchmark (primary key to extract)
BENCHMARK_METRICS = {
    "gqa": "exact_match,none",
    "scienceqa": "exact_match,none",
    "realworldqa": "exact_match,flexible-extract",
    "chartqa": "relaxed_overall,none",
    "mmstar": "average,none",
    "pope": "pope_accuracy,none",
    "vstar_bench": "vstar_overall_acc,none",
    "mmvet": "gpt_eval_score,none",
    "llava_in_the_wild": "gpt_eval_llava_all,none",
}

# Benchmarks on 0-100 scale (excluded from 0-1 average)
SCALE_100_BENCHMARKS = {"mmvet", "vstar_bench", "llava_in_the_wild"}

# Fallback: try common metric patterns if the primary one isn't found
METRIC_FALLBACKS = [
    "exact_match,none",
    "exact_match,flexible-extract",
    "relaxed_overall,none",
    "relaxed_accuracy,none",
    "average,none",
    "pope_accuracy,none",
    "vstar_overall_acc,none",
    "gpt_eval_score,none",
    "accuracy,none",
]


def find_results_json(output_dir: Path) -> Path | None:
    """Find the most recent *_results.json in an output directory (searching subdirs)."""
    candidates = sorted(output_dir.rglob("*_results.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def extract_task_score(results_data: dict, task_name: str) -> float | None:
    """Extract the primary metric score for a task from results JSON."""
    results = results_data.get("results", {})

    # Try the task directly
    task_data = results.get(task_name, {})
    if not task_data:
        # Try case-insensitive match
        for key in results:
            if key.lower() == task_name.lower():
                task_data = results[key]
                break

    if not task_data:
        return None

    # Try the expected metric first
    expected = BENCHMARK_METRICS.get(task_name)
    if expected and expected in task_data:
        return task_data[expected]

    # Fallback: try common metric patterns
    for metric_key in METRIC_FALLBACKS:
        if metric_key in task_data:
            return task_data[metric_key]

    return None


def load_all_results(output_dir: Path) -> dict:
    """Load all result JSONs from an output directory and merge task scores."""
    scores = {}
    for results_file in output_dir.rglob("*_results.json"):
        with open(results_file) as f:
            data = json.load(f)
        for task_name in BENCHMARK_METRICS:
            if task_name not in scores:
                score = extract_task_score(data, task_name)
                if score is not None:
                    scores[task_name] = score
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compare Base vs MoDA evaluation results")
    parser.add_argument("--base", type=str, default="eval_results/base", help="Base model results dir")
    parser.add_argument("--moda", type=str, default="eval_results/moda", help="MoDA model results dir")
    args = parser.parse_args()

    base_dir = Path(args.base)
    moda_dir = Path(args.moda)

    base_scores = load_all_results(base_dir)
    moda_scores = load_all_results(moda_dir)

    # Print comparison table
    print()
    print("=" * 72)
    print(f"{'Benchmark':<15} {'Base':>10} {'MoDA':>10} {'Delta':>10} {'Δ%':>8}")
    print("=" * 72)

    benchmarks = list(BENCHMARK_METRICS.keys())
    for bench in benchmarks:
        base_val = base_scores.get(bench)
        moda_val = moda_scores.get(bench)

        base_str = f"{base_val:.4f}" if base_val is not None else "---"
        moda_str = f"{moda_val:.4f}" if moda_val is not None else "---"

        if base_val is not None and moda_val is not None:
            delta = moda_val - base_val
            delta_pct = (delta / base_val * 100) if base_val != 0 else 0
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.4f}"
            pct_str = f"{sign}{delta_pct:.1f}%"
        else:
            delta_str = "---"
            pct_str = "---"

        print(f"{bench:<15} {base_str:>10} {moda_str:>10} {delta_str:>10} {pct_str:>8}")

    print("=" * 72)

    # Averages (only for 0-1 scale benchmarks, excluding 0-100 scale ones)
    scale01_benchmarks = [b for b in benchmarks if b not in SCALE_100_BENCHMARKS]
    paired = [(base_scores[b], moda_scores[b]) for b in scale01_benchmarks if b in base_scores and b in moda_scores]
    if paired:
        avg_base = sum(b for b, _ in paired) / len(paired)
        avg_moda = sum(m for _, m in paired) / len(paired)
        avg_delta = avg_moda - avg_base
        sign = "+" if avg_delta >= 0 else ""
        pct = (avg_delta / avg_base * 100) if avg_base != 0 else 0
        print(f"{'Avg (0-1)':<15} {avg_base:>10.4f} {avg_moda:>10.4f} {sign}{avg_delta:>9.4f} {sign}{pct:>7.1f}%")
        print(f"  (over {len(paired)} benchmarks, excluding mmvet)")
    print()


if __name__ == "__main__":
    main()
