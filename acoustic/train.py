"""Compare leak-safe classical models and save the best validation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
METADATA = {"anon_id", "label", "split"}


def score(y_true, probability) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted)),
    }


def make_models(k: int) -> dict[str, Pipeline]:
    common = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
    ]
    return {
        "logistic": Pipeline(common + [("model", LogisticRegression(C=0.2, max_iter=5000, class_weight="balanced", random_state=42))]),
        "rbf_svm": Pipeline(common + [("model", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42))]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)),
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=HERE / "features.csv")
    parser.add_argument("--model-dir", type=Path, default=HERE / "model")
    parser.add_argument("--feature-count", type=int, default=120)
    args = parser.parse_args()
    data = pd.read_csv(args.features)
    names = [column for column in data if column not in METADATA]
    k = min(args.feature_count, len(names))
    train, val = data[data.split == "train"], data[data.split == "val"]
    y_train = (train.label == "synthetic").astype(int).to_numpy()
    y_val = (val.label == "synthetic").astype(int).to_numpy()
    results, fitted = {}, {}
    for name, model in make_models(k).items():
        model.fit(train[names], y_train)
        results[name] = score(y_val, model.predict_proba(val[names])[:, 1])
        fitted[name] = model
        print(name, json.dumps(results[name], indent=2))
    winner = max(results, key=lambda name: results[name]["roc_auc"])
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[winner], args.model_dir / "acoustic_model.joblib")
    (args.model_dir / "training_report.json").write_text(json.dumps({"winner": winner, "validation": results, "features": names}, indent=2))
    print(f"Saved {winner} to {args.model_dir / 'acoustic_model.joblib'}")


if __name__ == "__main__":
    main()
