import csv
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt

from deep_audio.preprocessing import (
    HOP_LENGTH,
    FMAX,
    load_caller_audio,
    segment_audio,
    select_high_energy_segment,
    audio_to_log_mel,
)


MANIFEST_PATH = Path("manifest.csv")
AUDIO_DIR = Path("audio")


def find_example(label):
    """
    Find one training example with the requested label.
    """

    with open(
        MANIFEST_PATH,
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if (
                row["label"] == label
                and row["split"] == "train"
            ):
                return row["anon_id"]

    raise ValueError(
        f"No training example found for label: {label}"
    )


def prepare_example(anon_id):
    """
    Load caller audio, segment it, select the highest-energy
    segment, and convert that segment to Log-Mel.
    """

    audio_path = (
        AUDIO_DIR /
        f"{anon_id}.wav"
    )

    caller, sr = load_caller_audio(
        audio_path
    )

    segments = segment_audio(
        caller
    )

    (
        selected_segment,
        segment_index,
        rms
    ) = select_high_energy_segment(
        segments
    )

    log_mel = audio_to_log_mel(
        selected_segment
    )

    return {
        "id": anon_id,
        "sr": sr,
        "segment_index": segment_index,
        "rms": rms,
        "log_mel": log_mel
    }


human_id = find_example(
    "human"
)

synthetic_id = find_example(
    "synthetic"
)


human = prepare_example(
    human_id
)

synthetic = prepare_example(
    synthetic_id
)


print("\n=== HUMAN ===")
print(
    "Call:",
    human["id"]
)
print(
    "Selected segment:",
    human["segment_index"]
)
print(
    "RMS:",
    human["rms"]
)

print("\n=== SYNTHETIC ===")
print(
    "Call:",
    synthetic["id"]
)
print(
    "Selected segment:",
    synthetic["segment_index"]
)
print(
    "RMS:",
    synthetic["rms"]
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8)
)


human_img = librosa.display.specshow(
    human["log_mel"],
    sr=human["sr"],
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="mel",
    fmax=FMAX,
    ax=axes[0]
)

axes[0].set_title(
    f"Human — {human['id']} "
    f"(segment {human['segment_index']})"
)

fig.colorbar(
    human_img,
    ax=axes[0],
    format="%+2.0f dB"
)


synthetic_img = librosa.display.specshow(
    synthetic["log_mel"],
    sr=synthetic["sr"],
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="mel",
    fmax=FMAX,
    ax=axes[1]
)

axes[1].set_title(
    f"Synthetic — {synthetic['id']} "
    f"(segment {synthetic['segment_index']})"
)

fig.colorbar(
    synthetic_img,
    ax=axes[1],
    format="%+2.0f dB"
)


plt.tight_layout()
plt.show()