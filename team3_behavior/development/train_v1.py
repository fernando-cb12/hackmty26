"""Train an explainable logistic-regression behavior baseline.

This implementation uses only NumPy so the first experiment stays small and
transparent. It trains on manifest split=train and reports split=val exactly
once as the held-out hackathon validation set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from ..feature_extraction import FEATURE_NAMES


def load_split(path: Path, split: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == split]
    x = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows])
    y = np.asarray([1.0 if row["label"] == "synthetic" else 0.0 for row in rows])
    return rows, x, y


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    """Fit weighted logistic regression with the Adam optimizer."""
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    positive_weight = len(y) / (2.0 * max(float(y.sum()), 1.0))
    negative_weight = len(y) / (2.0 * max(float((1.0 - y).sum()), 1.0))
    sample_weights = np.where(y == 1.0, positive_weight, negative_weight)

    mw = np.zeros_like(weights)
    vw = np.zeros_like(weights)
    mb = vb = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for step in range(1, epochs + 1):
        probabilities = sigmoid(x @ weights + bias)
        error = sample_weights * (probabilities - y)
        grad_w = (x.T @ error) / len(y) + l2 * weights
        grad_b = float(error.mean())

        mw = beta1 * mw + (1.0 - beta1) * grad_w
        vw = beta2 * vw + (1.0 - beta2) * (grad_w * grad_w)
        mb = beta1 * mb + (1.0 - beta1) * grad_b
        vb = beta2 * vb + (1.0 - beta2) * (grad_b * grad_b)
        mw_hat = mw / (1.0 - beta1**step)
        vw_hat = vw / (1.0 - beta2**step)
        mb_hat = mb / (1.0 - beta1**step)
        vb_hat = vb / (1.0 - beta2**step)
        weights -= learning_rate * mw_hat / (np.sqrt(vw_hat) + epsilon)
        bias -= learning_rate * mb_hat / (math.sqrt(vb_hat) + epsilon)
    return weights, bias


def auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(probabilities)
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1, dtype=float)
    positives = labels == 1.0
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def metrics(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    positive = labels == 1.0
    tp = int(np.sum(predictions & positive))
    tn = int(np.sum(~predictions & ~positive))
    fp = int(np.sum(predictions & ~positive))
    fn = int(np.sum(~predictions & positive))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr
    return {
        "calls": len(labels),
        "threshold": threshold,
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "accuracy": (tp + tn) / len(labels),
        "precision_synthetic": precision,
        "recall_synthetic": recall,
        "f1_synthetic": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": auc(probabilities, labels),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.05)
    args = parser.parse_args()

    train_rows, train_x, train_y = load_split(args.features, "train")
    val_rows, val_x, val_y = load_split(args.features, "val")
    feature_mean = train_x.mean(axis=0)
    feature_std = train_x.std(axis=0)
    feature_std[feature_std < 1e-9] = 1.0
    train_scaled = (train_x - feature_mean) / feature_std
    val_scaled = (val_x - feature_mean) / feature_std

    started = time.perf_counter()
    weights, bias = train_logistic_regression(
        train_scaled, train_y, args.epochs, args.learning_rate, args.l2
    )
    training_seconds = time.perf_counter() - started
    train_probabilities = sigmoid(train_scaled @ weights + bias)
    val_started = time.perf_counter()
    val_probabilities = sigmoid(val_scaled @ weights + bias)
    inference_seconds = time.perf_counter() - val_started

    train_metrics = metrics(train_probabilities, train_y)
    val_metrics = metrics(val_probabilities, val_y)
    model = {
        "model_type": "weighted_logistic_regression",
        "probability_meaning": "probability that caller is synthetic",
        "threshold": 0.5,
        "feature_names": FEATURE_NAMES,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "training": {
            "train_calls": len(train_rows),
            "val_calls": len(val_rows),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "training_seconds": training_seconds,
            "mean_val_inference_ms": inference_seconds * 1000.0 / len(val_rows),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(model, indent=2), encoding="utf-8")

    if args.predictions_out:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_out.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["anon_id", "label", "probability_synthetic", "prediction"],
            )
            writer.writeheader()
            for row, probability in zip(val_rows, val_probabilities):
                writer.writerow(
                    {
                        "anon_id": row["anon_id"],
                        "label": row["label"],
                        "probability_synthetic": float(probability),
                        "prediction": "synthetic" if probability >= 0.5 else "human",
                    }
                )

    print(f"model written to {args.model_out}")
    print(json.dumps({"train": train_metrics, "val": val_metrics}, indent=2))


if __name__ == "__main__":
    main()
