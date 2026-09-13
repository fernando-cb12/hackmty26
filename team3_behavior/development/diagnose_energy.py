"""Diagnose caller VAD false positives as possible echo, noise, or speech.

This is an investigative tool, not part of model inference. It compares the
energy VAD against Altur's reference turns and summarizes every extra caller
interval with energy, agent overlap, channel-envelope correlation and spectral
flatness indicators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from ..feature_extraction import load_turns
from ..vad import VadConfig, detect_turns, read_stereo_wav


def frame_mask(turns: list[dict], channel: int, frame_count: int, frame_s: float) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=bool)
    for turn in turns:
        if turn["channel"] != channel:
            continue
        start = max(0, int(math.floor(turn["start"] / frame_s)))
        end = min(frame_count, int(math.ceil(turn["end"] / frame_s)))
        mask[start:end] = True
    return mask


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    result = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < len(mask) and mask[end]:
            end += 1
        result.append((index, end))
        index = end
    return result


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spectral_flatness(framed: np.ndarray) -> np.ndarray:
    window = np.hanning(framed.shape[1]).astype(np.float32)
    power = np.abs(np.fft.rfft(framed * window, axis=1)) ** 2 + 1e-12
    geometric = np.exp(np.mean(np.log(power), axis=1))
    arithmetic = np.mean(power, axis=1)
    return geometric / arithmetic


def classify_interval(agent_overlap: float, correlation: float, agent_louder_db: float, flatness: float) -> str:
    """Return a cautious heuristic label, never a ground-truth claim."""
    if agent_overlap >= 0.70 and correlation >= 0.45 and agent_louder_db >= 3.0:
        return "possible_echo"
    if flatness >= 0.28 and correlation < 0.35:
        return "possible_noise"
    if agent_overlap <= 0.25 and flatness < 0.28:
        return "possible_unlabeled_speech"
    return "ambiguous"


def diagnose(audio_path: Path, official_turns_path: Path, config: VadConfig = VadConfig()) -> dict:
    samples, sample_rate = read_stereo_wav(audio_path)
    predicted_turns, vad_diagnostics = detect_turns(samples, sample_rate, config)
    official_turns = load_turns(official_turns_path)
    frame_samples = round(sample_rate * config.frame_ms / 1000)
    frame_count = int(math.ceil(len(samples) / frame_samples))
    padded = np.zeros((frame_count * frame_samples, 2), dtype=np.float32)
    padded[: len(samples)] = samples
    framed = padded.reshape(frame_count, frame_samples, 2)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    energy_db = 20.0 * np.log10(rms + 1e-12)
    caller_flatness = spectral_flatness(framed[:, :, 0])
    frame_s = config.frame_ms / 1000.0

    caller_official = frame_mask(official_turns, 0, frame_count, frame_s)
    agent_official = frame_mask(official_turns, 1, frame_count, frame_s)
    caller_predicted = frame_mask(predicted_turns, 0, frame_count, frame_s)
    caller_extra = caller_predicted & ~caller_official

    intervals = []
    for start, end in true_runs(caller_extra):
        selected = slice(start, end)
        overlap = float(np.mean(agent_official[selected]))
        correlation = safe_correlation(energy_db[selected, 0], energy_db[selected, 1])
        caller_energy = float(np.mean(energy_db[selected, 0]))
        agent_energy = float(np.mean(energy_db[selected, 1]))
        flatness = float(np.median(caller_flatness[selected]))
        intervals.append(
            {
                "start_s": round(start * frame_s, 3),
                "end_s": round(end * frame_s, 3),
                "duration_s": round((end - start) * frame_s, 3),
                "agent_overlap_ratio": round(overlap, 4),
                "energy_correlation": round(correlation, 4),
                "caller_mean_dbfs": round(caller_energy, 3),
                "agent_mean_dbfs": round(agent_energy, 3),
                "agent_louder_db": round(agent_energy - caller_energy, 3),
                "caller_spectral_flatness": round(flatness, 4),
                "hypothesis": classify_interval(
                    overlap, correlation, agent_energy - caller_energy, flatness
                ),
            }
        )

    total_extra = float(np.sum(caller_extra) * frame_s)
    extra_with_agent = float(np.sum(caller_extra & agent_official) * frame_s)
    hypothesis_seconds: dict[str, float] = {}
    for interval in intervals:
        label = interval["hypothesis"]
        hypothesis_seconds[label] = hypothesis_seconds.get(label, 0.0) + interval["duration_s"]

    return {
        "call_id": audio_path.stem,
        "duration_s": vad_diagnostics["duration_s"],
        "frame_ms": config.frame_ms,
        "caller_threshold_dbfs": vad_diagnostics["channels"][0]["threshold_db"],
        "agent_threshold_dbfs": vad_diagnostics["channels"][1]["threshold_db"],
        "caller_extra_s": round(total_extra, 3),
        "extra_coincident_with_official_agent_s": round(extra_with_agent, 3),
        "extra_coincident_with_official_agent_ratio": round(
            extra_with_agent / total_extra if total_extra else 0.0, 4
        ),
        "hypothesis_seconds": {
            key: round(value, 3) for key, value in sorted(hypothesis_seconds.items())
        },
        "intervals": intervals,
        "energy_dbfs": {
            "time_s": np.round(np.arange(frame_count) * frame_s, 3).tolist(),
            "caller": np.round(energy_db[:, 0], 2).tolist(),
            "agent": np.round(energy_db[:, 1], 2).tolist(),
            "caller_official": caller_official.astype(np.uint8).tolist(),
            "agent_official": agent_official.astype(np.uint8).tolist(),
            "caller_predicted": caller_predicted.astype(np.uint8).tolist(),
            "caller_extra": caller_extra.astype(np.uint8).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--official-turns", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    report = diagnose(args.wav, args.official_turns)
    compact = {key: value for key, value in report.items() if key != "energy_dbfs"}
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", encoding="utf-8", newline="") as stream:
            fieldnames = list(report["intervals"][0]) if report["intervals"] else []
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(report["intervals"])


if __name__ == "__main__":
    main()
