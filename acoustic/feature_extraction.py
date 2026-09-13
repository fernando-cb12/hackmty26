"""Acoustic feature extraction for the HackMTY Altur challenge.

Features are calculated from caller (channel 0) speech turns only. The turn
annotations prevent long conversational silences from dominating voice features.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

STATS = ("mean", "std", "p10", "median", "p90")


def calculate_statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Summarise frames, replacing non-finite values with safe values."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return {f"{prefix}_{name}": 0.0 for name in STATS}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def load_caller_audio(wav_path: str | Path) -> tuple[np.ndarray, int]:
    """Load the caller channel from a challenge stereo WAV."""
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(f"Expected stereo WAV, received shape {audio.shape} from {wav_path}")
    return audio[:, 0], sample_rate


def load_caller_audio_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load the caller channel from in-memory judge WAV bytes."""
    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(f"Expected stereo WAV, received shape {audio.shape} from byte payload")
    return audio[:, 0], sample_rate

def load_caller_turns(turns_path: str | Path, sample_rate: int, n_samples: int) -> list[tuple[int, int]]:
    with open(turns_path, encoding="utf-8") as handle:
        turns = json.load(handle)["turns"]
    clips = []
    for turn in turns:
        if turn["channel"] != 0 or turn["end"] - turn["start"] < 0.15:
            continue
        start = max(0, int(round(turn["start"] * sample_rate)))
        end = min(n_samples, int(round(turn["end"] * sample_rate)))
        if end - start >= int(0.15 * sample_rate):
            clips.append((start, end))
    return clips


def _feature_frames(signal: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    """Extract framewise voice features from one speech clip."""
    n_fft, hop = 512, 80  # 64 ms / 10 ms frames at 8 kHz
    frames: dict[str, np.ndarray] = {}
    frames["rms"] = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop)[0]
    frames["zcr"] = librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop)[0]
    frames["centroid"] = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop)[0]
    frames["bandwidth"] = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop)[0]
    frames["rolloff"] = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop, roll_percent=0.85)[0]
    frames["flatness"] = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft, hop_length=hop)[0]

    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    for kind, matrix in (("mfcc", mfcc), ("delta", librosa.feature.delta(mfcc)), ("delta2", librosa.feature.delta(mfcc, order=2))):
        for index, values in enumerate(matrix, start=1):
            frames[f"{kind}{index:02d}"] = values

    mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=16, n_fft=n_fft, hop_length=hop, fmax=3800)
    for index, values in enumerate(librosa.power_to_db(mel, ref=np.max), start=1):
        frames[f"mel{index:02d}"] = values

    # YIN is fast enough for all 353 calls. Filter silence-like frames.
    f0 = librosa.yin(signal, fmin=65, fmax=380, sr=sample_rate, frame_length=n_fft, hop_length=hop)
    rms = frames["rms"]
    frames["f0"] = f0[rms >= max(np.percentile(rms, 20), 1e-4)]
    return frames


def extract_features(caller_audio: np.ndarray, sample_rate: int, clips: Iterable[tuple[int, int]] | None = None) -> dict[str, float]:
    """Create one fixed-size feature vector from a caller's audio."""
    if clips is None:
        clips = [(0, len(caller_audio))]
    all_frames: dict[str, list[np.ndarray]] = {}
    durations = []
    for start, end in clips:
        signal = np.asarray(caller_audio[start:end], dtype=np.float32)
        if len(signal) < 512:
            continue
        signal = signal - np.mean(signal)  # Safe DC removal; no peak normalisation.
        durations.append(len(signal) / sample_rate)
        for name, values in _feature_frames(signal, sample_rate).items():
            all_frames.setdefault(name, []).append(values)
    if not all_frames:
        raise ValueError("No usable caller speech was found")
    features: dict[str, float] = {}
    for name, arrays in all_frames.items():
        features.update(calculate_statistics(np.concatenate(arrays), name))
    features["caller_speech_seconds"] = float(sum(durations))
    features["caller_turn_count"] = float(len(durations))
    features.update(calculate_statistics(np.asarray(durations), "caller_turn_duration"))
    return features


def extract_call_features(wav_path: str | Path, turns_path: str | Path | None = None) -> dict[str, float]:
    caller, sample_rate = load_caller_audio(wav_path)
    clips = load_caller_turns(turns_path, sample_rate, len(caller)) if turns_path else None
    return extract_features(caller, sample_rate, clips)


def clips_from_turns(turns: Iterable[dict], sample_rate: int, n_samples: int) -> list[tuple[int, int]]:
    clips = []
    for turn in turns:
        if turn.get("channel") != 0 or turn["end"] - turn["start"] < 0.15:
            continue
        start = max(0, int(round(turn["start"] * sample_rate)))
        end = min(n_samples, int(round(turn["end"] * sample_rate)))
        if end - start >= int(0.15 * sample_rate):
            clips.append((start, end))
    return clips


def extract_call_features_from_bytes(wav_bytes: bytes, turns: Iterable[dict] | None = None) -> dict[str, float]:
    caller, sample_rate = load_caller_audio_bytes(wav_bytes)
    clips = clips_from_turns(turns, sample_rate, len(caller)) if turns is not None else None
    return extract_features(caller, sample_rate, clips)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print acoustic features for one call.")
    parser.add_argument("wav_path")
    parser.add_argument("--turns")
    args = parser.parse_args()
    for name, value in extract_call_features(args.wav_path, args.turns).items():
        print(f"{name}: {value:.6f}")
