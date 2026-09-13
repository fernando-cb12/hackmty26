"""Train, cross-validate, and evaluate the semantic caller detector."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:  # Supports both `python team4_semantic/train.py` and package imports.
    from .feature_extraction import SentenceEmbedder, extract_features
    from .transcribe import transcribe_call
except ImportError:  # pragma: no cover - direct-script compatibility
    from feature_extraction import SentenceEmbedder, extract_features
    from transcribe import transcribe_call


def _feature_rows(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [record["features"] for record in records]


def _feature_pipeline(calibrated: bool = False) -> Pipeline:
    from sklearn.feature_extraction import DictVectorizer

    classifier: Any = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.7)
    if calibrated:
        classifier = CalibratedClassifierCV(classifier, method="sigmoid", cv=5)
    return Pipeline([("vectorize", DictVectorizer(sparse=False)), ("scale", StandardScaler()), ("classifier", classifier)])


def _text_pipeline(calibrated: bool = False) -> Pipeline:
    classifier: Any = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.8)
    if calibrated:
        classifier = CalibratedClassifierCV(classifier, method="sigmoid", cv=5)
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=12000, sublinear_tf=True)),
        ("classifier", classifier),
    ])


def _best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.2, 0.8, 121)
    return float(max(candidates, key=lambda threshold: f1_score(labels, probabilities >= threshold)))


def _oof_scores(records: list[dict[str, Any]], labels: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {"features": np.zeros(len(records)), "text": np.zeros(len(records))}
    texts = [record["caller_text"] for record in records]
    features = _feature_rows(records)
    for train_idx, test_idx in splits.split(records, labels):
        feature_model, text_model = _feature_pipeline(), _text_pipeline()
        feature_model.fit([features[i] for i in train_idx], labels[train_idx])
        text_model.fit([texts[i] for i in train_idx], labels[train_idx])
        scores["features"][test_idx] = feature_model.predict_proba([features[i] for i in test_idx])[:, 1]
        scores["text"][test_idx] = text_model.predict_proba([texts[i] for i in test_idx])[:, 1]
    scores["average"] = (scores["features"] + scores["text"]) / 2
    metrics = {name: roc_auc_score(labels, values) for name, values in scores.items()}
    return scores, metrics


def _load_records(dataset_root: Path, split: str, cache_dir: Path, device: str) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    with (dataset_root / "manifest.csv").open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == split]
    embedder = SentenceEmbedder()
    records: list[dict[str, Any]] = []
    labels: list[int] = []
    identifiers: list[str] = []
    for index, row in enumerate(rows, 1):
        audio = dataset_root / "audio" / f"{row['anon_id']}.wav"
        turns = dataset_root / "turns" / f"{row['anon_id']}.json"
        utterances = transcribe_call(audio, turns, cache_dir, device=device)
        records.append({
            "features": extract_features(utterances, embedder),
            "caller_text": " ".join(item["text"] for item in utterances if item["channel"] == 0),
        })
        labels.append(int(row["label"] == "synthetic"))
        identifiers.append(row["anon_id"])
        print(f"[{split} {index}/{len(rows)}] {row['anon_id']}")
    return records, np.asarray(labels), identifiers


def _metric_report(labels: np.ndarray, probabilities: np.ndarray, threshold: float, identifiers: list[str] | None = None) -> dict[str, Any]:
    predictions = probabilities >= threshold
    report = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    if identifiers is not None:
        report["false_positives"] = [call_id for call_id, label, prediction in zip(identifiers, labels, predictions) if not label and prediction]
        report["false_negatives"] = [call_id for call_id, label, prediction in zip(identifiers, labels, predictions) if label and not prediction]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the offline semantic caller detector.")
    parser.add_argument("--dataset-root", default=str(Path(__file__).parent.parent))
    parser.add_argument("--cache-dir", default=str(Path(__file__).parent / "cache"))
    parser.add_argument("--model-dir", default=str(Path(__file__).parent / "model"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    root, model_dir = Path(args.dataset_root), Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train, y_train, _ = _load_records(root, "train", Path(args.cache_dir), args.device)
    oof, oof_auc = _oof_scores(train, y_train)
    # Require a material OOF improvement before adding the less interpretable text model.
    best_base = max("features", "text", key=lambda name: oof_auc[name])
    selected = "average" if oof_auc["average"] >= oof_auc[best_base] + 0.005 else best_base
    threshold = _best_threshold(y_train, oof[selected])
    feature_model = _feature_pipeline(calibrated=True).fit(_feature_rows(train), y_train)
    text_model = _text_pipeline(calibrated=True).fit([item["caller_text"] for item in train], y_train)
    bundle = {"selected_model": selected, "threshold": threshold, "feature_model": feature_model, "text_model": text_model, "feature_schema": sorted(train[0]["features"]), "oof_roc_auc": oof_auc}
    joblib.dump(bundle, model_dir / "semantic_model.joblib")

    validation_started = time.perf_counter()
    validation, y_val, ids = _load_records(root, "val", Path(args.cache_dir), args.device)
    validation_seconds = time.perf_counter() - validation_started
    feature_probability = feature_model.predict_proba(_feature_rows(validation))[:, 1]
    text_probability = text_model.predict_proba([item["caller_text"] for item in validation])[:, 1]
    probability = {"features": feature_probability, "text": text_probability, "average": (feature_probability + text_probability) / 2}[selected]
    report = {
        "model": selected,
        "train_oof_roc_auc": oof_auc,
        "validation": _metric_report(y_val, probability, threshold, ids),
        "validation_transcription_and_inference_seconds": validation_seconds,
        "validation_mean_seconds_per_call": validation_seconds / len(validation),
        "training_and_validation_seconds": time.perf_counter() - started,
    }
    (model_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (model_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["anon_id", "label", "probability", "prediction"])
        writer.writeheader()
        for call_id, label, score in zip(ids, y_val, probability):
            writer.writerow({"anon_id": call_id, "label": int(label), "probability": float(score), "prediction": int(score >= threshold)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
