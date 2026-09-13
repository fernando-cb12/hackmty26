"""Fast, explainable voice-activity detection for stereo telephone WAV files.

The detector uses adaptive energy thresholds independently for both channels.
Its public output matches Altur's turn format so another VAD implementation
(for example WebRTC VAD) can replace it later without changing downstream code.
"""

from __future__ import annotations

import argparse
import io
import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VadConfig:
    frame_ms: int = 20
    min_speech_ms: int = 160
    merge_gap_ms: int = 300
    padding_ms: int = 80
    noise_percentile: float = 20.0
    speech_percentile: float = 90.0
    threshold_fraction: float = 0.35
    minimum_snr_db: float = 6.0
    echo_suppression: bool = True
    echo_margin_db: float = 18.0


def _read_stereo_wav(source) -> tuple[np.ndarray, int]:
    with wave.open(source, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        payload = wav.readframes(frame_count)
    if channels != 2 or sample_width != 2 or sample_rate != 8000:
        raise ValueError(
            f"expected stereo 16-bit 8 kHz WAV, got "
            f"{channels} channels, {sample_width * 8}-bit, {sample_rate} Hz"
        )
    samples = np.frombuffer(payload, dtype="<i2").reshape(-1, 2)
    return samples.astype(np.float32) / 32768.0, sample_rate


def read_stereo_wav(path: Path) -> tuple[np.ndarray, int]:
    return _read_stereo_wav(str(path))


def _fill_short_false_runs(activity: np.ndarray, maximum_frames: int) -> np.ndarray:
    result = activity.copy()
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        end = index
        while end < len(result) and not result[end]:
            end += 1
        bounded_by_speech = index > 0 and end < len(result)
        if bounded_by_speech and end - index <= maximum_frames:
            result[index:end] = True
        index = end
    return result


def _remove_short_true_runs(activity: np.ndarray, minimum_frames: int) -> np.ndarray:
    result = activity.copy()
    index = 0
    while index < len(result):
        if not result[index]:
            index += 1
            continue
        end = index
        while end < len(result) and result[end]:
            end += 1
        if end - index < minimum_frames:
            result[index:end] = False
        index = end
    return result


def _pad_activity(activity: np.ndarray, padding_frames: int) -> np.ndarray:
    if padding_frames <= 0 or not activity.any():
        return activity.copy()
    kernel = np.ones(2 * padding_frames + 1, dtype=np.int16)
    return np.convolve(activity.astype(np.int16), kernel, mode="same") > 0


def _smooth_activity(activity: np.ndarray, config: VadConfig) -> np.ndarray:
    frame_ms = config.frame_ms
    gap_frames = max(0, round(config.merge_gap_ms / frame_ms))
    speech_frames = max(1, round(config.min_speech_ms / frame_ms))
    padding_frames = max(0, round(config.padding_ms / frame_ms))
    result = _fill_short_false_runs(activity, gap_frames)
    result = _remove_short_true_runs(result, speech_frames)
    result = _pad_activity(result, padding_frames)
    return _fill_short_false_runs(result, gap_frames)


def _activity_to_turns(
    activity: np.ndarray,
    channel: int,
    frame_seconds: float,
    duration_seconds: float,
) -> list[dict]:
    turns = []
    index = 0
    while index < len(activity):
        if not activity[index]:
            index += 1
            continue
        end = index
        while end < len(activity) and activity[end]:
            end += 1
        turns.append(
            {
                "channel": channel,
                "start": round(index * frame_seconds, 3),
                "end": round(min(end * frame_seconds, duration_seconds), 3),
            }
        )
        index = end
    return turns


def detect_turns(
    samples: np.ndarray,
    sample_rate: int,
    config: VadConfig = VadConfig(),
) -> tuple[list[dict], dict]:
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must have shape (sample_count, 2)")
    frame_samples = round(sample_rate * config.frame_ms / 1000)
    if frame_samples <= 0:
        raise ValueError("frame_ms is too small")
    duration_seconds = len(samples) / sample_rate
    frame_count = int(np.ceil(len(samples) / frame_samples))
    padded = np.zeros((frame_count * frame_samples, 2), dtype=np.float32)
    padded[: len(samples)] = samples
    framed = padded.reshape(frame_count, frame_samples, 2)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    energy_db = 20.0 * np.log10(rms + 1e-12)

    thresholds = np.zeros(2, dtype=float)
    raw_activity = np.zeros_like(energy_db, dtype=bool)
    channel_stats = []
    for channel in range(2):
        values = energy_db[:, channel]
        noise_db = float(np.percentile(values, config.noise_percentile))
        speech_db = float(np.percentile(values, config.speech_percentile))
        dynamic_range = max(0.0, speech_db - noise_db)
        threshold_db = noise_db + max(
            config.minimum_snr_db, config.threshold_fraction * dynamic_range
        )
        if dynamic_range > 3.0:
            threshold_db = min(threshold_db, speech_db - 3.0)
        thresholds[channel] = threshold_db
        raw_activity[:, channel] = values >= threshold_db
        channel_stats.append(
            {
                "noise_db": round(noise_db, 3),
                "speech_reference_db": round(speech_db, 3),
                "threshold_db": round(threshold_db, 3),
            }
        )

    echo_frames_removed = [0, 0]
    if config.echo_suppression:
        simultaneous = raw_activity[:, 0] & raw_activity[:, 1]
        difference = energy_db[:, 0] - energy_db[:, 1]
        suppress_caller = simultaneous & (difference <= -config.echo_margin_db)
        suppress_agent = simultaneous & (difference >= config.echo_margin_db)
        raw_activity[suppress_caller, 0] = False
        raw_activity[suppress_agent, 1] = False
        echo_frames_removed = [int(suppress_caller.sum()), int(suppress_agent.sum())]

    activity = np.column_stack(
        [_smooth_activity(raw_activity[:, channel], config) for channel in range(2)]
    )
    frame_seconds = config.frame_ms / 1000.0
    turns = []
    for channel in range(2):
        turns.extend(
            _activity_to_turns(
                activity[:, channel], channel, frame_seconds, duration_seconds
            )
        )
        channel_stats[channel]["speech_ratio"] = round(float(activity[:, channel].mean()), 6)
        channel_stats[channel]["turn_count"] = int(
            np.sum(activity[:, channel] & ~np.r_[False, activity[:-1, channel]])
        )
        channel_stats[channel]["echo_frames_removed"] = echo_frames_removed[channel]
    turns.sort(key=lambda turn: (turn["start"], turn["channel"], turn["end"]))
    diagnostics = {
        "duration_s": round(duration_seconds, 3),
        "config": asdict(config),
        "channels": channel_stats,
    }
    return turns, diagnostics


def detect_turns_from_wav(
    path: Path, config: VadConfig = VadConfig()
) -> tuple[list[dict], dict]:
    samples, sample_rate = read_stereo_wav(path)
    return detect_turns(samples, sample_rate, config)


def detect_turns_from_wav_bytes(
    payload: bytes, config: VadConfig = VadConfig()
) -> tuple[list[dict], dict]:
    samples, sample_rate = _read_stereo_wav(io.BytesIO(payload))
    return detect_turns(samples, sample_rate, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-echo-suppression", action="store_true")
    args = parser.parse_args()
    config = VadConfig(echo_suppression=not args.no_echo_suppression)
    turns, diagnostics = detect_turns_from_wav(args.wav, config)
    payload = {"turns": turns, "vad": diagnostics}
    encoded = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"wrote {len(turns)} turns to {args.output}")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
