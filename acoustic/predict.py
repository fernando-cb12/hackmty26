"""Predict whether a single stereo call contains a synthetic caller."""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

try:
    from .feature_extraction import extract_call_features, extract_call_features_from_bytes
except ImportError:  # Allows direct script execution.
    from feature_extraction import extract_call_features, extract_call_features_from_bytes


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "model" / "acoustic_model.joblib"


@lru_cache(maxsize=4)
def load_model(model_path: str | Path = DEFAULT_MODEL):
    return joblib.load(model_path)


def _score_features(features: dict[str, float], model) -> float:
    row = np.array([[features.get(name, 0.0) for name in model.feature_names_in_]], dtype=float)
    return float(model.predict_proba(row)[0, 1])


def _result(probability: float, elapsed_ms: float | None = None) -> dict:
    is_synthetic = probability >= 0.5
    confidence = probability if is_synthetic else 1.0 - probability
    result = {
        "is_synthetic": bool(is_synthetic),
        "probability_synthetic": probability,
        "confidence": confidence,
    }
    if elapsed_ms is not None:
        result["elapsed_ms"] = elapsed_ms
    return result


def predict(audio_path: str | Path, turns_path: str | Path | None = None, model_path: str | Path = DEFAULT_MODEL) -> dict:
    started = time.perf_counter()
    model = load_model(model_path)
    features = extract_call_features(audio_path, turns_path)
    result = _result(_score_features(features, model), (time.perf_counter() - started) * 1000.0)
    result["probability"] = result["probability_synthetic"]
    return result


def predict_wav_bytes(
    wav_bytes: bytes,
    turns: list[dict] | None = None,
    model_path: str | Path = DEFAULT_MODEL,
) -> dict:
    """Return acoustic synthetic probability from judge WAV bytes."""
    started = time.perf_counter()
    model = load_model(model_path)
    features = extract_call_features_from_bytes(wav_bytes, turns)
    return _result(_score_features(features, model), (time.perf_counter() - started) * 1000.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--turns")
    parser.add_argument("--model", default=HERE / "model" / "acoustic_model.joblib")
    args = parser.parse_args()
    print(predict(args.audio_path, args.turns, args.model))
