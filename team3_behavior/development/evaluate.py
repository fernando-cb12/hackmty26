"""Evaluate the complete WAV -> VAD -> features -> prediction pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..predict import DEFAULT_MODEL, load_model, predict_wav
from .train_v1 import metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--n", type=int, default=0, help="0 evaluates the complete split")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == args.split]
    if args.n:
        rows = rows[: args.n]

    probabilities = []
    labels = []
    latencies_ms = []
    for index, row in enumerate(rows, 1):
        result = predict_wav(args.audio_dir / f"{row['anon_id']}.wav", args.model)
        probabilities.append(result["probability_synthetic"])
        labels.append(1.0 if row["label"] == "synthetic" else 0.0)
        latencies_ms.append(result["vad_summary"]["processing_ms"])
        if index % 25 == 0 or index == len(rows):
            print(f"evaluated {index}/{len(rows)} calls", flush=True)

    threshold = float(load_model(args.model).get("threshold", 0.5))
    report = metrics(np.asarray(probabilities), np.asarray(labels), threshold)
    report["mean_processing_ms"] = float(np.mean(latencies_ms))
    report["max_processing_ms"] = float(np.max(latencies_ms))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
