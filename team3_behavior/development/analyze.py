"""Compare behavior-feature distributions for human and synthetic calls."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from ..feature_extraction import FEATURE_NAMES


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def standardized_difference(human: list[float], synthetic: list[float]) -> float:
    """Difference in means expressed in pooled standard deviations."""
    if len(human) < 2 or len(synthetic) < 2:
        return 0.0
    human_var = statistics.variance(human)
    synthetic_var = statistics.variance(synthetic)
    pooled = math.sqrt(
        ((len(human) - 1) * human_var + (len(synthetic) - 1) * synthetic_var)
        / (len(human) + len(synthetic) - 2)
    )
    return (mean(synthetic) - mean(human)) / pooled if pooled > 1e-12 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    with args.features.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.split != "all":
        rows = [row for row in rows if row["split"] == args.split]

    human_rows = [row for row in rows if row["label"] == "human"]
    synthetic_rows = [row for row in rows if row["label"] == "synthetic"]
    comparisons = []
    for feature in FEATURE_NAMES:
        human = [float(row[feature]) for row in human_rows]
        synthetic = [float(row[feature]) for row in synthetic_rows]
        effect = standardized_difference(human, synthetic)
        comparisons.append((abs(effect), effect, feature, mean(human), mean(synthetic)))

    comparisons.sort(reverse=True)
    print(
        f"split={args.split} calls={len(rows)} "
        f"human={len(human_rows)} synthetic={len(synthetic_rows)}"
    )
    print("\nLargest univariate differences (exploration, not model accuracy):")
    print(f"{'feature':32s} {'human mean':>12s} {'synthetic':>12s} {'effect':>9s}")
    for _, effect, feature, human_mean, synthetic_mean in comparisons[: args.top]:
        print(f"{feature:32s} {human_mean:12.4f} {synthetic_mean:12.4f} {effect:9.3f}")


if __name__ == "__main__":
    main()
