"""Run aligned validation predictions for endpoint candidates.

This is intended for the stronger machine. It measures each candidate on the
same calls, records latency, and reports whether simple behavior+acoustic
fusion should remain the live endpoint choice.

Example:
    python scripts/ensemble_experiment.py --split val --n 0 --out reports/ensemble_val.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acoustic.predict import predict_wav_bytes as predict_acoustic_bytes
from team3_behavior.predict import predict_from_turns
from team3_behavior.vad import detect_turns_from_wav_bytes


def auc(scores: list[float], labels: list[int]) -> float | None:
    pairs = sorted(zip(scores, labels))
    ranks: dict[int, float] = {}
    index = 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        for rank_index in range(index, end):
            ranks[rank_index] = (index + end + 1) / 2
        index = end
    pos = [ranks[i] for i, (_, label) in enumerate(pairs) if label == 1]
    n_pos = len(pos)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def score_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [row for row in rows if "error" not in row]
    labels = [1 if row["label"] == "synthetic" else 0 for row in answered]
    probs = [float(row["probability_synthetic"]) for row in answered]
    preds = [1 if prob >= 0.5 else 0 for prob in probs]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    latencies = [float(row["elapsed_ms"]) / 1000.0 for row in answered if row.get("elapsed_ms") is not None]
    return {
        "calls": len(rows),
        "answered": len(answered),
        "errors": len(rows) - len(answered),
        "balanced_accuracy": ((tp / n_pos) + (tn / n_neg)) / 2 if n_pos and n_neg else None,
        "accuracy": (tp + tn) / len(answered) if answered else None,
        "auc": auc(probs, labels) if answered else None,
        "brier": sum((prob - label) ** 2 for prob, label in zip(probs, labels)) / len(answered) if answered else None,
        "confusion_matrix_human_synthetic": [[tn, fp], [fn, tp]],
        "false_positives": [row["call_id"] for row, y, p in zip(answered, labels, preds) if y == 0 and p == 1],
        "false_negatives": [row["call_id"] for row, y, p in zip(answered, labels, preds) if y == 1 and p == 0],
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "p95_latency_s": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else None),
        "max_latency_s": max(latencies) if latencies else None,
    }


def load_manifest(path: Path, split: str, limit: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if split != "all":
        rows = [row for row in rows if row["split"] == split]
    rows.sort(key=lambda row: row["anon_id"])
    return rows[:limit] if limit else rows


def try_import_deep() -> Callable[[bytes], dict[str, Any]] | None:
    try:
        from deep_audio.predict import predict_wav_bytes
    except Exception as error:
        print(f"deep_audio unavailable: {error}", file=sys.stderr)
        return None
    return predict_wav_bytes


def try_import_semantic() -> Callable[[Path, Path | None], dict[str, Any]] | None:
    try:
        from team4_semantic.predict import predict
    except Exception as error:
        print(f"semantic unavailable: {error}", file=sys.stderr)
        return None
    return predict


def normalize_result(result: dict[str, Any], elapsed_ms: float | None = None) -> dict[str, Any]:
    probability = result.get("probability_synthetic", result.get("probability"))
    if probability is None:
        raise ValueError("candidate did not return probability_synthetic or probability")
    probability = float(probability)
    is_synthetic = probability >= 0.5
    confidence = probability if is_synthetic else 1.0 - probability
    return {
        "probability_synthetic": probability,
        "is_synthetic": bool(is_synthetic),
        "confidence": float(result.get("confidence", confidence)),
        "elapsed_ms": float(result.get("elapsed_ms", elapsed_ms or 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.csv")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "audio")
    parser.add_argument("--turns-dir", type=Path, default=ROOT / "turns")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--n", type=int, default=0, help="number of calls, 0 = all")
    parser.add_argument("--include-deep", action="store_true")
    parser.add_argument("--include-semantic", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "ensemble_experiment.json")
    args = parser.parse_args()

    rows = load_manifest(args.manifest, args.split, args.n)
    deep_predict = try_import_deep() if args.include_deep else None
    semantic_predict = try_import_semantic() if args.include_semantic else None

    candidate_rows: dict[str, list[dict[str, Any]]] = {
        "behavior": [],
        "acoustic": [],
        "behavior_acoustic_average": [],
    }
    if deep_predict:
        candidate_rows["deep_audio"] = []
    if semantic_predict:
        candidate_rows["semantic"] = []

    for index, row in enumerate(rows, start=1):
        call_id = row["anon_id"]
        label = row["label"]
        audio_path = args.audio_dir / f"{call_id}.wav"
        wav_bytes = audio_path.read_bytes()
        print(f"[{index:03d}/{len(rows):03d}] {call_id}", flush=True)

        try:
            started = time.perf_counter()
            turns, diagnostics = detect_turns_from_wav_bytes(wav_bytes)
            behavior = normalize_result(
                predict_from_turns(turns, diagnostics, started=started),
                (time.perf_counter() - started) * 1000.0,
            )
            acoustic = normalize_result(predict_acoustic_bytes(wav_bytes, turns=turns))
            average_prob = (behavior["probability_synthetic"] + acoustic["probability_synthetic"]) / 2
            average = normalize_result(
                {
                    "probability_synthetic": average_prob,
                    "elapsed_ms": behavior["elapsed_ms"] + acoustic["elapsed_ms"],
                }
            )
            for name, result in (
                ("behavior", behavior),
                ("acoustic", acoustic),
                ("behavior_acoustic_average", average),
            ):
                candidate_rows[name].append({"call_id": call_id, "label": label, **result})
        except Exception as error:
            for name in ("behavior", "acoustic", "behavior_acoustic_average"):
                candidate_rows[name].append({"call_id": call_id, "label": label, "error": str(error)})

        if deep_predict:
            try:
                candidate_rows["deep_audio"].append({"call_id": call_id, "label": label, **normalize_result(deep_predict(wav_bytes))})
            except Exception as error:
                candidate_rows["deep_audio"].append({"call_id": call_id, "label": label, "error": str(error)})

        if semantic_predict:
            try:
                turns_path = args.turns_dir / f"{call_id}.json"
                started = time.perf_counter()
                result = semantic_predict(audio_path, turns_path if turns_path.exists() else None, device="cpu")
                candidate_rows["semantic"].append(
                    {
                        "call_id": call_id,
                        "label": label,
                        **normalize_result(result, (time.perf_counter() - started) * 1000.0),
                    }
                )
            except Exception as error:
                candidate_rows["semantic"].append({"call_id": call_id, "label": label, "error": str(error)})

    summary = {name: score_predictions(predictions) for name, predictions in candidate_rows.items()}
    error_sets = {
        name: sorted(
            row["call_id"]
            for row in predictions
            if "error" in row
            or (float(row["probability_synthetic"]) >= 0.5) != (row["label"] == "synthetic")
        )
        for name, predictions in candidate_rows.items()
    }
    output = {"summary": summary, "error_overlap": error_sets, "predictions": candidate_rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
