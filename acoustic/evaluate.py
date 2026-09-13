"""Produce detailed validation metrics for a saved acoustic model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix, roc_auc_score


HERE = Path(__file__).resolve().parent
METADATA = {"anon_id", "label", "split"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=HERE / "features.csv")
    parser.add_argument("--model", type=Path, default=HERE / "model" / "acoustic_model.joblib")
    parser.add_argument("--output-dir", type=Path, default=HERE / "model")
    args = parser.parse_args()
    data = pd.read_csv(args.features)
    val = data[data.split == "val"]
    names = [column for column in data if column not in METADATA]
    y_true = (val.label == "synthetic").astype(int).to_numpy()
    probability = joblib.load(args.model).predict_proba(val[names])[:, 1]
    predicted = (probability >= 0.5).astype(int)
    report = {
        "calls": len(val),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix_labels_human_synthetic": confusion_matrix(y_true, predicted).tolist(),
        "classification_report": classification_report(y_true, predicted, target_names=["human", "synthetic"], output_dict=True, zero_division=0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({
        "anon_id": val["anon_id"].to_numpy(),
        "label": val["label"].to_numpy(),
        "synthetic_probability": probability,
        "is_synthetic": predicted.astype(bool),
    }).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    print(f"Saved report and {len(val)} validation predictions to {args.output_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
