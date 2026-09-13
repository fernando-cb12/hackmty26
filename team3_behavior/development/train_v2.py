"""Compare V2 behavior classifiers with train-only cross-validation.

Model selection is based exclusively on five-fold cross-validation inside the
train split. The provided val split is reported only as a final check.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..feature_extraction import FEATURE_NAMES_V2
from .train_v1 import metrics


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def matrix(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows])
    y = np.asarray([1 if row["label"] == "synthetic" else 0 for row in rows])
    return x, y


def candidates() -> list[tuple[str, object, dict]]:
    return [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced", max_iter=5000, random_state=42
                        ),
                    ),
                ]
            ),
            {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=350,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [2, 5],
                "max_features": ["sqrt"],
            },
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                class_weight="balanced", max_iter=250, random_state=42
            ),
            {
                "learning_rate": [0.04, 0.08],
                "max_leaf_nodes": [7, 15],
                "min_samples_leaf": [10, 20],
                "l2_regularization": [0.5],
            },
        ),
    ]


def select_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    candidates = np.linspace(0.20, 0.80, 121)
    scored = [(metrics(probabilities, labels, float(value)), float(value)) for value in candidates]
    best_metrics, best_threshold = max(
        scored,
        key=lambda item: (
            item[0]["balanced_accuracy"],
            -abs(item[1] - 0.5),
        ),
    )
    return best_threshold, best_metrics


def json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.features)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    feature_sets = {
        "behavior_v2": [name for name in FEATURE_NAMES_V2 if not name.startswith("vad_")],
        "behavior_plus_vad_quality": list(FEATURE_NAMES_V2),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted = {}

    for feature_set_name, feature_names in feature_sets.items():
        train_x, train_y = matrix(train_rows, feature_names)
        val_x, val_y = matrix(val_rows, feature_names)
        for model_name, estimator, parameter_grid in candidates():
            started = time.perf_counter()
            search = GridSearchCV(
                estimator,
                parameter_grid,
                scoring="balanced_accuracy",
                cv=cv,
                n_jobs=-1,
                refit=True,
                return_train_score=True,
            )
            search.fit(train_x, train_y)
            elapsed = time.perf_counter() - started
            best_index = search.best_index_
            val_probabilities = search.best_estimator_.predict_proba(val_x)[:, 1]
            result = {
                "feature_set": feature_set_name,
                "feature_count": len(feature_names),
                "model": model_name,
                "best_parameters": search.best_params_,
                "cv_balanced_accuracy_mean": search.best_score_,
                "cv_balanced_accuracy_std": search.cv_results_["std_test_score"][best_index],
                "cv_train_balanced_accuracy_mean": search.cv_results_[
                    "mean_train_score"
                ][best_index],
                "val_metrics": metrics(val_probabilities, val_y),
                "search_seconds": elapsed,
            }
            results.append(result)
            key = (feature_set_name, model_name)
            fitted[key] = (search.best_estimator_, feature_names, val_probabilities)
            print(
                f"{feature_set_name:27s} {model_name:24s} "
                f"CV={search.best_score_:.4f} "
                f"val={result['val_metrics']['balanced_accuracy']:.4f} "
                f"errors={result['val_metrics']['fp'] + result['val_metrics']['fn']}",
                flush=True,
            )

    # This choice deliberately ignores val performance.
    winner_result = max(results, key=lambda item: item["cv_balanced_accuracy_mean"])
    winner_key = (winner_result["feature_set"], winner_result["model"])
    winner_estimator, winner_features, winner_probabilities = fitted[winner_key]
    winner_train_x, winner_train_y = matrix(train_rows, winner_features)
    winner_val_x, winner_val_y = matrix(val_rows, winner_features)
    out_of_fold_probabilities = cross_val_predict(
        winner_estimator,
        winner_train_x,
        winner_train_y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold, threshold_cv_metrics = select_threshold(
        out_of_fold_probabilities, winner_train_y
    )
    winner_probabilities = winner_estimator.predict_proba(winner_val_x)[:, 1]
    winner_result["selected_threshold"] = threshold
    winner_result["threshold_selection_cv_metrics"] = threshold_cv_metrics
    winner_result["val_metrics_selected_threshold"] = metrics(
        winner_probabilities, winner_val_y, threshold
    )
    artifact = {
        "model_type": "sklearn_behavior_v2",
        "probability_meaning": "probability that caller is synthetic",
        "threshold": threshold,
        "feature_names": winner_features,
        "estimator": winner_estimator,
        "selection": json_ready(winner_result),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_out)

    report = {
        "selection_rule": "highest mean balanced accuracy in 5-fold train-only CV",
        "winner": json_ready(winner_result),
        "all_results": json_ready(
            sorted(results, key=lambda item: item["cv_balanced_accuracy_mean"], reverse=True)
        ),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["anon_id", "label", "probability_synthetic", "prediction"],
        )
        writer.writeheader()
        for row, probability in zip(val_rows, winner_probabilities):
            writer.writerow(
                {
                    "anon_id": row["anon_id"],
                    "label": row["label"],
                    "probability_synthetic": float(probability),
                    "prediction": "synthetic" if probability >= threshold else "human",
                }
            )
    print(f"winner saved to {args.model_out}")
    print(json.dumps(report["winner"], indent=2))


if __name__ == "__main__":
    main()
