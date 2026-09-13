import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

from deep_audio.preprocessing import (
    load_caller_audio,
    segment_audio,
)

from deep_audio.vad import speech_ratio


MANIFEST_PATH = Path("manifest.csv")
AUDIO_DIR = Path("audio")


with open(MANIFEST_PATH, newline="") as f:
    rows = list(csv.DictReader(f))


records = []


for i, row in enumerate(rows, start=1):

    call_id = row["anon_id"]
    label = row["label"]
    split = row["split"]

    audio_path = AUDIO_DIR / f"{call_id}.wav"

    caller, sr = load_caller_audio(audio_path)

    segments = segment_audio(caller)

    for segment_index, segment in enumerate(segments):

        ratio = speech_ratio(
            segment,
            sr=sr
        )

        records.append({
            "call_id": call_id,
            "label": label,
            "split": split,
            "segment_index": segment_index,
            "speech_ratio": ratio,
        })

    print(
        f"[{i:03d}/{len(rows)}] "
        f"{call_id} -> {len(segments)} segments"
    )


all_ratios = np.array(
    [record["speech_ratio"] for record in records],
    dtype=np.float32
)


print("\n=== GLOBAL SPEECH RATIO ===")

print("Total calls:", len(rows))
print("Total segments:", len(records))

print(
    "Zero activity:",
    np.mean(all_ratios == 0)
)

print(
    "Below 10%:",
    np.mean(all_ratios < 0.10)
)

print(
    "Below 20%:",
    np.mean(all_ratios < 0.20)
)

print(
    "Below 30%:",
    np.mean(all_ratios < 0.30)
)

print("\nDistribution:")

for p in [
    0,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    100,
]:
    print(
        f"P{p:02d}: "
        f"{np.percentile(all_ratios, p):.4f}"
    )


grouped = defaultdict(list)

for record in records:

    key = (
        record["split"],
        record["label"]
    )

    grouped[key].append(
        record["speech_ratio"]
    )


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
        float(np.percentile(values, 10))
    )

    print(
        "P90:",
        float(np.percentile(values, 90))
    )

    print(
        "Speech >= 20%:",
        float(np.mean(values >= 0.20))
    )

    print(
        "Speech >= 30%:",
        float(np.mean(values >= 0.30))
    )

    print(
        "Speech >= 40%:",
        float(np.mean(values >= 0.40))
    )