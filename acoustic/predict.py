"""Predict whether a single stereo call contains a synthetic caller."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from feature_extraction import extract_call_features


HERE = Path(__file__).resolve().parent


def predict(audio_path: str | Path, turns_path: str | Path | None = None, model_path: str | Path = HERE / "model" / "acoustic_model.joblib") -> dict:
    model = joblib.load(model_path)
    features = extract_call_features(audio_path, turns_path)
    row = pd.DataFrame([features]).reindex(columns=model.feature_names_in_, fill_value=0.0)
    probability = float(model.predict_proba(row)[0, 1])
    return {"is_synthetic": probability >= 0.5, "probability": probability}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--turns")
    parser.add_argument("--model", default=HERE / "model" / "acoustic_model.joblib")
    args = parser.parse_args()
    print(predict(args.audio_path, args.turns, args.model))
