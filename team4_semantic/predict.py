"""Inference interface for the trained semantic caller detector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib

try:  # Supports both `python team4_semantic/predict.py` and package imports.
    from .feature_extraction import extract_features
    from .transcribe import transcribe_call
except ImportError:  # pragma: no cover - direct-script compatibility
    from feature_extraction import extract_features
    from transcribe import transcribe_call


def predict(
    audio_path: str | Path,
    turns_path: str | Path | None = None,
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Return an independent semantic synthetic probability for a stereo WAV call."""
    started = time.perf_counter()
    model_file = Path(model_path) if model_path else Path(__file__).parent / "model" / "semantic_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"No trained semantic model at {model_file}. Run train.py first.")
    bundle = joblib.load(model_file)
    utterances = transcribe_call(audio_path, turns_path, cache_dir, device=device)
    features = extract_features(utterances)
    caller_text = " ".join(item["text"] for item in utterances if item["channel"] == 0)
    feature_probability = float(bundle["feature_model"].predict_proba([features])[0, 1])
    text_probability = float(bundle["text_model"].predict_proba([caller_text])[0, 1])
    probability = {"features": feature_probability, "text": text_probability, "average": (feature_probability + text_probability) / 2}[bundle["selected_model"]]
    return {
        "is_synthetic": bool(probability >= bundle["threshold"]),
        "probability": float(probability),
        "reasoning_features": features,
        "inference_time_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether a caller is synthetic from semantic content.")
    parser.add_argument("audio")
    parser.add_argument("--turns")
    parser.add_argument("--model")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    print(json.dumps(predict(args.audio, args.turns, args.model, args.cache_dir, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
