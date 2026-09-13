"""Call-level byte inference for the deep-audio CNN candidate."""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from deep_audio.model import DeepAudioCNN
from deep_audio.preprocessing import (
    audio_to_log_mel,
    load_caller_audio,
    load_caller_audio_bytes,
    peak_normalize,
    segment_audio,
)
from deep_audio.vad import speech_ratio


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "deep_audio_cnn_peaknorm.pt"
MIN_SPEECH_RATIO = 0.20


@lru_cache(maxsize=2)
def load_model(model_path: str | Path = DEFAULT_MODEL, device_name: str = "auto"):
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if device_name == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model = DeepAudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def _predict_caller_audio(caller: np.ndarray, sample_rate: int, model, device) -> tuple[float, int]:
    segments = segment_audio(caller, sr=sample_rate)
    tensors = []
    for segment in segments:
        if speech_ratio(segment, sr=sample_rate) < MIN_SPEECH_RATIO:
            continue
        log_mel = audio_to_log_mel(peak_normalize(segment), sr=sample_rate)
        tensors.append(torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0))
    if not tensors:
        raise ValueError("No speech-active caller segments were found")
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(batch)).detach().cpu().numpy()
    return float(np.mean(probabilities)), len(tensors)


def _result(probability: float, elapsed_ms: float, segment_count: int) -> dict:
    is_synthetic = probability >= 0.5
    confidence = probability if is_synthetic else 1.0 - probability
    return {
        "is_synthetic": bool(is_synthetic),
        "probability_synthetic": probability,
        "confidence": confidence,
        "elapsed_ms": elapsed_ms,
        "segment_count": segment_count,
    }


def predict_wav_bytes(
    wav_bytes: bytes,
    model_path: str | Path = DEFAULT_MODEL,
    device: str = "auto",
) -> dict:
    started = time.perf_counter()
    model, torch_device = load_model(model_path, device)
    caller, sample_rate = load_caller_audio_bytes(wav_bytes)
    probability, segment_count = _predict_caller_audio(caller, sample_rate, model, torch_device)
    return _result(probability, (time.perf_counter() - started) * 1000.0, segment_count)


def predict(
    audio_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL,
    device: str = "auto",
) -> dict:
    started = time.perf_counter()
    model, torch_device = load_model(model_path, device)
    caller, sample_rate = load_caller_audio(audio_path)
    probability, segment_count = _predict_caller_audio(caller, sample_rate, model, torch_device)
    return _result(probability, (time.perf_counter() - started) * 1000.0, segment_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    print(json.dumps(predict(args.audio, args.model, args.device), indent=2))


if __name__ == "__main__":
    main()
