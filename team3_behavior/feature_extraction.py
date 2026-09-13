"""Extract conversation-behavior features from two-channel speech turns.

Channel 0 is the caller being classified and channel 1 is the bank agent.
The module deliberately contains no machine-learning code: its output is a
plain table that can be inspected and used by different classifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


FEATURE_NAMES = [
    "call_duration_s",
    "caller_turn_count",
    "agent_turn_count",
    "caller_speech_s",
    "agent_speech_s",
    "caller_speech_ratio",
    "agent_speech_ratio",
    "caller_turn_mean_s",
    "caller_turn_std_s",
    "caller_turn_cv",
    "caller_turn_median_s",
    "caller_turn_p10_s",
    "caller_turn_p90_s",
    "agent_turn_mean_s",
    "agent_turn_std_s",
    "response_count",
    "response_latency_mean_s",
    "response_latency_std_s",
    "response_latency_cv",
    "response_latency_median_s",
    "response_latency_p90_s",
    "caller_pause_mean_s",
    "caller_pause_std_s",
    "caller_pause_cv",
    "overlap_event_count",
    "overlap_total_s",
    "overlap_per_minute",
    "caller_overlap_ratio",
    "caller_interrupts_agent",
    "agent_interrupts_caller",
    "caller_interrupt_rate",
    "agent_interrupt_rate",
]

FEATURE_NAMES_V2 = FEATURE_NAMES + [
    "caller_turn_iqr_s",
    "caller_turn_mad_s",
    "caller_short_turn_ratio",
    "caller_long_turn_ratio",
    "caller_duration_delta_mean_s",
    "caller_turns_per_minute",
    "agent_turns_per_minute",
    "speech_balance_ratio",
    "response_latency_iqr_s",
    "response_latency_mad_s",
    "response_under_0_5_ratio",
    "response_under_1_ratio",
    "response_under_2_ratio",
    "response_delta_abs_mean_s",
    "response_latency_trend_s_per_response",
    "clean_response_ratio",
    "response_early_mean_s",
    "response_middle_mean_s",
    "response_late_mean_s",
    "response_late_minus_early_s",
    "caller_speech_early_ratio",
    "caller_speech_middle_ratio",
    "caller_speech_late_ratio",
    "overlap_event_mean_s",
    "overlap_event_p90_s",
    "caller_interrupt_overlap_mean_s",
    "agent_interrupt_overlap_mean_s",
    "vad_caller_dynamic_range_db",
    "vad_agent_dynamic_range_db",
    "vad_caller_echo_removed_per_minute",
    "vad_agent_echo_removed_per_minute",
]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _cv(values: list[float]) -> float:
    mean = _mean(values)
    return _std(values) / mean if mean > 1e-9 else 0.0


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    median = _percentile(values, 0.50)
    return _percentile([abs(value - median) for value in values], 0.50)


def _ratio_below(values: list[float], limit: float) -> float:
    return sum(value < limit for value in values) / len(values) if values else 0.0


def _mean_absolute_delta(values: list[float]) -> float:
    deltas = [abs(current - previous) for previous, current in zip(values, values[1:])]
    return _mean(deltas)


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator <= 1e-12:
        return 0.0
    y_mean = _mean(values)
    return sum(
        (index - x_mean) * (value - y_mean) for index, value in enumerate(values)
    ) / denominator


def _interval_coverage(turns: list[dict], start: float, end: float) -> float:
    if end <= start:
        return 0.0
    covered = sum(
        max(0.0, min(turn["end"], end) - max(turn["start"], start)) for turn in turns
    )
    return covered / (end - start)


def _percentile(values: list[float], percentile: float) -> float:
    """Linearly interpolated percentile without third-party dependencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _active_at(turns: list[dict], timestamp: float) -> bool:
    return any(turn["start"] < timestamp < turn["end"] for turn in turns)


def _overlap_segments(caller: list[dict], agent: list[dict]) -> list[tuple[float, float]]:
    """Return intersections between sorted caller and agent intervals."""
    overlaps: list[tuple[float, float]] = []
    i = j = 0
    while i < len(caller) and j < len(agent):
        start = max(caller[i]["start"], agent[j]["start"])
        end = min(caller[i]["end"], agent[j]["end"])
        if start < end:
            overlaps.append((start, end))
        if caller[i]["end"] <= agent[j]["end"]:
            i += 1
        else:
            j += 1
    return overlaps


def _validate_turns(turns: Iterable[dict]) -> list[dict]:
    cleaned = []
    for index, turn in enumerate(turns):
        channel = turn.get("channel")
        start = float(turn.get("start", -1))
        end = float(turn.get("end", -1))
        if channel not in (0, 1):
            raise ValueError(f"turn {index}: channel must be 0 or 1")
        if start < 0 or end <= start:
            raise ValueError(f"turn {index}: invalid interval {start}..{end}")
        cleaned.append({"channel": channel, "start": start, "end": end})
    return sorted(cleaned, key=lambda turn: (turn["start"], turn["end"]))


def extract_features(turns: Iterable[dict], duration_s: float | None = None) -> dict[str, float]:
    """Create one conversation-level feature vector from speech intervals."""
    cleaned = _validate_turns(turns)
    caller = [turn for turn in cleaned if turn["channel"] == 0]
    agent = [turn for turn in cleaned if turn["channel"] == 1]

    inferred_duration = max((turn["end"] for turn in cleaned), default=0.0)
    duration = max(float(duration_s or 0.0), inferred_duration, 1e-9)
    caller_durations = [turn["end"] - turn["start"] for turn in caller]
    agent_durations = [turn["end"] - turn["start"] for turn in agent]
    caller_speech = sum(caller_durations)
    agent_speech = sum(agent_durations)

    # Latency is measured only when a caller turn begins after an agent turn has
    # ended. Starts occurring while the agent is active are interruptions.
    response_latencies: list[float] = []
    agent_ends = sorted(turn["end"] for turn in agent)
    for caller_turn in caller:
        previous_ends = [end for end in agent_ends if end <= caller_turn["start"]]
        if previous_ends and not _active_at(agent, caller_turn["start"]):
            response_latencies.append(caller_turn["start"] - previous_ends[-1])

    caller_pauses = [
        max(0.0, current["start"] - previous["end"])
        for previous, current in zip(caller, caller[1:])
    ]
    overlaps = _overlap_segments(caller, agent)
    overlap_total = sum(end - start for start, end in overlaps)
    caller_interrupts = sum(_active_at(agent, turn["start"]) for turn in caller)
    agent_interrupts = sum(_active_at(caller, turn["start"]) for turn in agent)
    minutes = duration / 60.0

    features = {
        "call_duration_s": duration,
        "caller_turn_count": float(len(caller)),
        "agent_turn_count": float(len(agent)),
        "caller_speech_s": caller_speech,
        "agent_speech_s": agent_speech,
        "caller_speech_ratio": caller_speech / duration,
        "agent_speech_ratio": agent_speech / duration,
        "caller_turn_mean_s": _mean(caller_durations),
        "caller_turn_std_s": _std(caller_durations),
        "caller_turn_cv": _cv(caller_durations),
        "caller_turn_median_s": _percentile(caller_durations, 0.50),
        "caller_turn_p10_s": _percentile(caller_durations, 0.10),
        "caller_turn_p90_s": _percentile(caller_durations, 0.90),
        "agent_turn_mean_s": _mean(agent_durations),
        "agent_turn_std_s": _std(agent_durations),
        "response_count": float(len(response_latencies)),
        "response_latency_mean_s": _mean(response_latencies),
        "response_latency_std_s": _std(response_latencies),
        "response_latency_cv": _cv(response_latencies),
        "response_latency_median_s": _percentile(response_latencies, 0.50),
        "response_latency_p90_s": _percentile(response_latencies, 0.90),
        "caller_pause_mean_s": _mean(caller_pauses),
        "caller_pause_std_s": _std(caller_pauses),
        "caller_pause_cv": _cv(caller_pauses),
        "overlap_event_count": float(len(overlaps)),
        "overlap_total_s": overlap_total,
        "overlap_per_minute": len(overlaps) / minutes,
        "caller_overlap_ratio": overlap_total / caller_speech if caller_speech else 0.0,
        "caller_interrupts_agent": float(caller_interrupts),
        "agent_interrupts_caller": float(agent_interrupts),
        "caller_interrupt_rate": caller_interrupts / len(caller) if caller else 0.0,
        "agent_interrupt_rate": agent_interrupts / len(agent) if agent else 0.0,
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}


def extract_features_v2(
    turns: Iterable[dict],
    duration_s: float | None = None,
    vad_diagnostics: dict | None = None,
) -> dict[str, float]:
    """Add robust, temporal and VAD-quality signals to the V1 feature set."""
    cleaned = _validate_turns(turns)
    base = extract_features(cleaned, duration_s)
    caller = [turn for turn in cleaned if turn["channel"] == 0]
    agent = [turn for turn in cleaned if turn["channel"] == 1]
    inferred_duration = max((turn["end"] for turn in cleaned), default=0.0)
    duration = max(float(duration_s or 0.0), inferred_duration, 1e-9)
    minutes = duration / 60.0
    caller_durations = [turn["end"] - turn["start"] for turn in caller]

    response_pairs: list[tuple[float, float]] = []
    agent_ends = sorted(turn["end"] for turn in agent)
    for caller_turn in caller:
        previous_ends = [end for end in agent_ends if end <= caller_turn["start"]]
        if previous_ends and not _active_at(agent, caller_turn["start"]):
            response_pairs.append(
                (caller_turn["start"], caller_turn["start"] - previous_ends[-1])
            )
    response_latencies = [latency for _, latency in response_pairs]

    thirds: list[list[float]] = [[], [], []]
    for timestamp, latency in response_pairs:
        third = min(2, int(3 * timestamp / duration))
        thirds[third].append(latency)
    third_means = [_mean(values) for values in thirds]
    third_duration = duration / 3.0
    caller_third_ratios = [
        _interval_coverage(caller, index * third_duration, (index + 1) * third_duration)
        for index in range(3)
    ]

    overlaps = _overlap_segments(caller, agent)
    overlap_durations = [end - start for start, end in overlaps]
    caller_interrupt_overlaps = []
    for turn in caller:
        active_agent_ends = [
            other["end"] for other in agent if other["start"] < turn["start"] < other["end"]
        ]
        if active_agent_ends:
            caller_interrupt_overlaps.append(min(active_agent_ends) - turn["start"])
    agent_interrupt_overlaps = []
    for turn in agent:
        active_caller_ends = [
            other["end"] for other in caller if other["start"] < turn["start"] < other["end"]
        ]
        if active_caller_ends:
            agent_interrupt_overlaps.append(min(active_caller_ends) - turn["start"])

    channels = (vad_diagnostics or {}).get("channels", [{}, {}])
    if len(channels) < 2:
        channels = [{}, {}]

    def dynamic_range(channel: int) -> float:
        stats = channels[channel]
        return max(
            0.0,
            float(stats.get("speech_reference_db", 0.0))
            - float(stats.get("noise_db", 0.0)),
        )

    total_speech = base["caller_speech_s"] + base["agent_speech_s"]
    v2 = {
        **base,
        "caller_turn_iqr_s": _percentile(caller_durations, 0.75)
        - _percentile(caller_durations, 0.25),
        "caller_turn_mad_s": _mad(caller_durations),
        "caller_short_turn_ratio": _ratio_below(caller_durations, 0.75),
        "caller_long_turn_ratio": sum(value > 5.0 for value in caller_durations)
        / len(caller_durations)
        if caller_durations
        else 0.0,
        "caller_duration_delta_mean_s": _mean_absolute_delta(caller_durations),
        "caller_turns_per_minute": len(caller) / minutes,
        "agent_turns_per_minute": len(agent) / minutes,
        "speech_balance_ratio": base["caller_speech_s"] / total_speech
        if total_speech
        else 0.0,
        "response_latency_iqr_s": _percentile(response_latencies, 0.75)
        - _percentile(response_latencies, 0.25),
        "response_latency_mad_s": _mad(response_latencies),
        "response_under_0_5_ratio": _ratio_below(response_latencies, 0.5),
        "response_under_1_ratio": _ratio_below(response_latencies, 1.0),
        "response_under_2_ratio": _ratio_below(response_latencies, 2.0),
        "response_delta_abs_mean_s": _mean_absolute_delta(response_latencies),
        "response_latency_trend_s_per_response": _linear_slope(response_latencies),
        "clean_response_ratio": len(response_latencies) / len(caller) if caller else 0.0,
        "response_early_mean_s": third_means[0],
        "response_middle_mean_s": third_means[1],
        "response_late_mean_s": third_means[2],
        "response_late_minus_early_s": third_means[2] - third_means[0],
        "caller_speech_early_ratio": caller_third_ratios[0],
        "caller_speech_middle_ratio": caller_third_ratios[1],
        "caller_speech_late_ratio": caller_third_ratios[2],
        "overlap_event_mean_s": _mean(overlap_durations),
        "overlap_event_p90_s": _percentile(overlap_durations, 0.90),
        "caller_interrupt_overlap_mean_s": _mean(caller_interrupt_overlaps),
        "agent_interrupt_overlap_mean_s": _mean(agent_interrupt_overlaps),
        "vad_caller_dynamic_range_db": dynamic_range(0),
        "vad_agent_dynamic_range_db": dynamic_range(1),
        "vad_caller_echo_removed_per_minute": float(
            channels[0].get("echo_frames_removed", 0.0)
        )
        * 0.02
        / minutes,
        "vad_agent_echo_removed_per_minute": float(
            channels[1].get("echo_frames_removed", 0.0)
        )
        * 0.02
        / minutes,
    }
    return {name: float(v2[name]) for name in FEATURE_NAMES_V2}


def load_turns(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload.get("turns"), list):
        raise ValueError(f"{path}: expected a 'turns' list")
    return payload["turns"]


def build_feature_table(
    manifest: Path, turns_dir: Path, feature_names: list[str] = FEATURE_NAMES
) -> list[dict]:
    with manifest.open(encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))

    output_rows = []
    for row in manifest_rows:
        call_id = row["anon_id"]
        turn_path = turns_dir / f"{call_id}.json"
        if not turn_path.exists():
            raise FileNotFoundError(f"missing turns for {call_id}: {turn_path}")
        turns = load_turns(turn_path)
        if feature_names == FEATURE_NAMES_V2:
            features = extract_features_v2(turns, float(row["duration_s"]))
        else:
            features = extract_features(turns, float(row["duration_s"]))
        output_rows.append(
            {"anon_id": call_id, "label": row["label"], "split": row["split"], **features}
        )
    return output_rows


def build_feature_table_from_audio(
    manifest: Path, audio_dir: Path, feature_names: list[str] = FEATURE_NAMES
) -> list[dict]:
    # Imported only for audio mode, keeping JSON-only feature extraction simple.
    try:
        from .vad import detect_turns_from_wav
    except ImportError:  # Allows direct script execution.
        from vad import detect_turns_from_wav

    with manifest.open(encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))

    output_rows = []
    for index, row in enumerate(manifest_rows, 1):
        call_id = row["anon_id"]
        audio_path = audio_dir / f"{call_id}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"missing audio for {call_id}: {audio_path}")
        turns, diagnostics = detect_turns_from_wav(audio_path)
        if feature_names == FEATURE_NAMES_V2:
            features = extract_features_v2(turns, diagnostics["duration_s"], diagnostics)
        else:
            features = extract_features(turns, diagnostics["duration_s"])
        output_rows.append(
            {"anon_id": call_id, "label": row["label"], "split": row["split"], **features}
        )
        if index % 25 == 0 or index == len(manifest_rows):
            print(f"processed {index}/{len(manifest_rows)} calls", flush=True)
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--turns-dir", type=Path)
    source.add_argument("--audio-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    feature_names = FEATURE_NAMES_V2 if args.feature_version == "v2" else FEATURE_NAMES

    if args.audio_dir:
        rows = build_feature_table_from_audio(args.manifest, args.audio_dir, feature_names)
    else:
        rows = build_feature_table(args.manifest, args.turns_dir, feature_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["anon_id", "label", "split", *feature_names]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} calls and {len(feature_names)} features to {args.output}")


if __name__ == "__main__":
    main()
