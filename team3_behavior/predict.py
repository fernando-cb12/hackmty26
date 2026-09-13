"""Predict synthetic-caller probability from a complete stereo WAV call."""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from pathlib import Path

try:
    from .feature_extraction import extract_features, extract_features_v2
    from .vad import detect_turns_from_wav, detect_turns_from_wav_bytes
except ImportError:  # Allows direct script execution.
    from feature_extraction import extract_features, extract_features_v2
    from vad import detect_turns_from_wav, detect_turns_from_wav_bytes


DEFAULT_MODEL = Path(__file__).parent / "model" / "behavior_v2.joblib"


@lru_cache(maxsize=4)
def load_model(path: Path = DEFAULT_MODEL) -> dict:
    if path.suffix == ".joblib":
        import joblib

        model = joblib.load(path)
    else:
        model = json.loads(path.read_text(encoding="utf-8"))
    required = {"feature_names", "threshold"}
    missing = required - model.keys()
    if missing:
        raise ValueError(f"model is missing fields: {sorted(missing)}")
    return model


def score_features(features: dict[str, float], model: dict) -> float:
    if "estimator" in model:
        row = [[features[name] for name in model["feature_names"]]]
        return float(model["estimator"].predict_proba(row)[0, 1])
    linear_score = float(model["bias"])
    for name, mean, std, weight in zip(
        model["feature_names"],
        model["feature_mean"],
        model["feature_std"],
        model["weights"],
    ):
        linear_score += ((features[name] - mean) / std) * weight
    linear_score = max(-35.0, min(35.0, linear_score))
    return 1.0 / (1.0 + math.exp(-linear_score))


def _build_result(
    turns: list[dict], diagnostics: dict, model: dict, started: float
) -> dict:
    if model.get("model_type") == "sklearn_behavior_v2":
        features = extract_features_v2(turns, diagnostics["duration_s"], diagnostics)
    else:
        features = extract_features(turns, diagnostics["duration_s"])
    probability = score_features(features, model)
    threshold = float(model.get("threshold", 0.5))
    is_synthetic = probability >= threshold
    confidence = probability if is_synthetic else 1.0 - probability
    channels = diagnostics["channels"]
    processing_ms = (time.perf_counter() - started) * 1000.0
    return {
        "is_synthetic": bool(is_synthetic),
        "probability_synthetic": round(probability, 6),
        "confidence": round(confidence, 6),
        "features": {name: round(value, 6) for name, value in features.items()},
        "vad_summary": {
            "duration_s": diagnostics["duration_s"],
            "caller_turns": channels[0]["turn_count"],
            "agent_turns": channels[1]["turn_count"],
            "caller_speech_ratio": channels[0]["speech_ratio"],
            "agent_speech_ratio": channels[1]["speech_ratio"],
            "caller_echo_frames_removed": channels[0]["echo_frames_removed"],
            "agent_echo_frames_removed": channels[1]["echo_frames_removed"],
            "processing_ms": round(processing_ms, 3),
        },
    }


def predict_wav(path: Path, model_path: Path = DEFAULT_MODEL) -> dict:
    started = time.perf_counter()
    model = load_model(model_path)
    turns, diagnostics = detect_turns_from_wav(path)
    return _build_result(turns, diagnostics, model, started)


def predict_wav_bytes(payload: bytes, model_path: Path = DEFAULT_MODEL) -> dict:
    """Entry point intended for the future HTTP /detect integration."""
    started = time.perf_counter()
    model = load_model(model_path)
    turns, diagnostics = detect_turns_from_wav_bytes(payload)
    return _build_result(turns, diagnostics, model, started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(predict_wav(args.wav, args.model), indent=2))


if __name__ == "__main__":
    main()
