"""Compare retrieval modes in a results file, with paired bootstrap confidence intervals.

Usage (from the project root):
    python -m scripts.compare_runs evaluation/results/20260920_230447_baseline.json
    python -m scripts.compare_runs evaluation/results/20260920_230506_typos.json --metric mrr@10
"""
import argparse
import json
import random
from itertools import combinations
from pathlib import Path

METRICS = ["recall@5", "recall@10", "mrr@10", "ndcg@10"]


def bootstrap_diff(a: list[float], b: list[float], rounds: int = 10000, seed: int = 0):
    """Mean of (a - b) with a 95% confidence interval, resampling the questions."""
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(rounds))
    return sum(diffs) / n, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare modes in a results file")
    parser.add_argument("results", type=Path)
    parser.add_argument("--metric", choices=METRICS, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    per_question = data["per_question"]
    modes = list(per_question)
    metrics = [args.metric] if args.metric else METRICS

    print(f"{args.results.name}: {data['num_questions']} questions")
    for metric in metrics:
        print(f"\n{metric}")
        for first, second in combinations(modes, 2):
            a = [row[metric] for row in per_question[first]]
            b = [row[metric] for row in per_question[second]]
            mean, low, high = bootstrap_diff(a, b)
            verdict = "CI excludes 0" if low > 0 or high < 0 else "CI includes 0 (within noise)"
            print(f"  {first} - {second}: {mean:+.3f}  95% CI [{low:+.3f}, {high:+.3f}]  {verdict}")


if __name__ == "__main__":
    main()