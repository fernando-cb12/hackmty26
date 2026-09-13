"""Compare the energy VAD with Altur's provided reference turns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..feature_extraction import load_turns
from ..vad import VadConfig, detect_turns_from_wav


def frame_mask(turns: list[dict], channel: int, frames: int, frame_s: float) -> np.ndarray:
    mask = np.zeros(frames, dtype=bool)
    for turn in turns:
        if turn["channel"] != channel:
            continue
        start = max(0, int(np.floor(turn["start"] / frame_s)))
        end = min(frames, int(np.ceil(turn["end"] / frame_s)))
        mask[start:end] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--turns-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--n", type=int, default=0, help="0 compares every matching call")
    parser.add_argument("--no-echo-suppression", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.split != "all":
        rows = [row for row in rows if row["split"] == args.split]
    if args.n:
        rows = rows[: args.n]

    config = VadConfig(echo_suppression=not args.no_echo_suppression)
    totals = {channel: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for channel in (0, 1)}
    for index, row in enumerate(rows, 1):
        call_id = row["anon_id"]
        predicted, diagnostics = detect_turns_from_wav(args.audio_dir / f"{call_id}.wav", config)
        reference = load_turns(args.turns_dir / f"{call_id}.json")
        frame_s = config.frame_ms / 1000.0
        frames = int(np.ceil(diagnostics["duration_s"] / frame_s))
        for channel in (0, 1):
            pred = frame_mask(predicted, channel, frames, frame_s)
            truth = frame_mask(reference, channel, frames, frame_s)
            totals[channel]["tp"] += int(np.sum(pred & truth))
            totals[channel]["tn"] += int(np.sum(~pred & ~truth))
            totals[channel]["fp"] += int(np.sum(pred & ~truth))
            totals[channel]["fn"] += int(np.sum(~pred & truth))
        if index % 25 == 0 or index == len(rows):
            print(f"compared {index}/{len(rows)} calls", flush=True)

    for channel in (0, 1):
        counts = totals[channel]
        tp, tn, fp, fn = (counts[name] for name in ("tp", "tn", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(
            f"channel {channel}: accuracy={accuracy:.3f} precision={precision:.3f} "
            f"recall={recall:.3f} f1={f1:.3f}"
        )


if __name__ == "__main__":
    main()
