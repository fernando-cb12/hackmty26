"""Build a call-level acoustic feature table from the challenge manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from feature_extraction import extract_call_features


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.csv")
    parser.add_argument("--audio-dir", type=Path, default=HERE / "audio")
    parser.add_argument("--turns-dir", type=Path, default=ROOT / "turns")
    parser.add_argument("--output", type=Path, default=HERE / "features.csv")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    # Extraction takes a few minutes. A checkpoint lets a stopped run continue
    # instead of recomputing finished calls.
    existing = pd.read_csv(args.output) if args.output.exists() else pd.DataFrame()
    completed = set(existing.get("anon_id", []))
    rows, failures = existing.to_dict("records"), []
    for number, call in enumerate(manifest.itertuples(index=False), start=1):
        if call.anon_id in completed:
            print(f"[{number}/{len(manifest)}] {call.anon_id} (cached)", flush=True)
            continue
        try:
            row = extract_call_features(
                args.audio_dir / f"{call.anon_id}.wav",
                args.turns_dir / f"{call.anon_id}.json",
            )
            row.update({"anon_id": call.anon_id, "label": call.label, "split": call.split})
            rows.append(row)
        except Exception as error:
            failures.append(f"{call.anon_id}: {error}")
        print(f"[{number}/{len(manifest)}] {call.anon_id}", flush=True)
        if number % 10 == 0 and rows:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values("anon_id").to_csv(args.output, index=False)
    if failures:
        print("Feature extraction failures:\n" + "\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    frame = pd.DataFrame(rows).sort_values("anon_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} calls x {len(frame.columns) - 3} features to {args.output}")


if __name__ == "__main__":
    main()
