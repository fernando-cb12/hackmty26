import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

from deep_audio.preprocessing import (
    load_caller_audio,
    segment_audio,
    rms_energy,
)


MANIFEST_PATH = Path("manifest.csv")
AUDIO_DIR = Path("audio")


def percentile(values, q):
    return float(np.percentile(values, q))


rows = []

with open(MANIFEST_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


all_records = []

for i, row in enumerate(rows, start=1):
    call_id = row["anon_id"]
    label = row["label"]
    split = row["split"]

    audio_path = AUDIO_DIR / f"{call_id}.wav"

    caller, sr = load_caller_audio(audio_path)
    segments = segment_audio(caller)

    for segment_index, segment in enumerate(segments):
        rms = rms_energy(segment)

        all_records.append({
            "call_id": call_id,
            "label": label,
            "split": split,
            "segment_index": segment_index,
            "rms": rms,
        })

    print(
        f"[{i:03d}/{len(rows)}] "
        f"{call_id} -> {len(segments)} segments"
    )


print("\n=== GLOBAL REPORT ===")

all_rms = np.array(
    [r["rms"] for r in all_records],
    dtype=np.float32
)

print("Total calls:", len(rows))
print("Total segments:", len(all_records))

print("\nRMS distribution:")
print("Min:", float(np.min(all_rms)))
print("P05:", percentile(all_rms, 5))
print("P10:", percentile(all_rms, 10))
print("P25:", percentile(all_rms, 25))
print("Median:", percentile(all_rms, 50))
print("P75:", percentile(all_rms, 75))
print("P90:", percentile(all_rms, 90))
print("P95:", percentile(all_rms, 95))
print("Max:", float(np.max(all_rms)))


grouped = defaultdict(list)

for record in all_records:
    grouped[
        (record["split"], record["label"])
    ].append(record["rms"])


print("\n=== BY SPLIT / LABEL ===")

for key in sorted(grouped):
    values = np.array(
        grouped[key],
        dtype=np.float32
    )

    split, label = key

    print(
        f"\n{split} / {label}"
    )
    print(
        "Segments:",
        len(values)
    )
    print(
        "Mean:",
        float(np.mean(values))
    )
    print(
        "Median:",
        float(np.median(values))
    )
    print(
        "P10:",
        percentile(values, 10)
    )
    print(
        "P90:",
        percentile(values, 90)
    )