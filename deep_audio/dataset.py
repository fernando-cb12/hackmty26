import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from deep_audio.preprocessing import (
    load_caller_audio,
    segment_audio,
    audio_to_log_mel,
    peak_normalize,
)

from deep_audio.preprocessing import (
    load_caller_audio,
    segment_audio,
    audio_to_log_mel,
)

from deep_audio.vad import speech_ratio


MANIFEST_PATH = Path("manifest.csv")
AUDIO_DIR = Path("audio")

MIN_SPEECH_RATIO = 0.20


class DeepAudioDataset(Dataset):
    def __init__(
        self,
        split,
        manifest_path=MANIFEST_PATH,
        audio_dir=AUDIO_DIR,
        min_speech_ratio=MIN_SPEECH_RATIO,
    ):
        self.split = split
        self.manifest_path = Path(manifest_path)
        self.audio_dir = Path(audio_dir)
        self.min_speech_ratio = min_speech_ratio

        self.samples = []

        self._build_index()

    def _build_index(self):
        """
        Scan calls in the requested split and store only
        segments with enough detected speech activity.
        """

        with open(self.manifest_path, newline="") as f:
            rows = list(csv.DictReader(f))

        rows = [
            row
            for row in rows
            if row["split"] == self.split
        ]

        print(
            f"Building {self.split} dataset "
            f"from {len(rows)} calls..."
        )

        for i, row in enumerate(rows, start=1):

            call_id = row["anon_id"]
            label_name = row["label"]

            label = 1.0 if label_name == "synthetic" else 0.0

            audio_path = (
                self.audio_dir /
                f"{call_id}.wav"
            )

            caller, sr = load_caller_audio(
                audio_path
            )

            segments = segment_audio(
                caller,
                sr=sr
            )

            kept = 0

            for segment_index, segment in enumerate(segments):

                ratio = speech_ratio(
                    segment,
                    sr=sr
                )

                if ratio < self.min_speech_ratio:
                    continue

                self.samples.append({
                    "call_id": call_id,
                    "audio_path": audio_path,
                    "segment_index": segment_index,
                    "label": label,
                })

                kept += 1

            print(
                f"[{i:03d}/{len(rows)}] "
                f"{call_id}: "
                f"{kept}/{len(segments)} kept"
            )

        print(
            f"\n{self.split.upper()} samples: "
            f"{len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        caller, sr = load_caller_audio(
            sample["audio_path"]
        )

        segments = segment_audio(
            caller,
            sr=sr
        )

        segment = segments[
            sample["segment_index"]
        ]

        segment = peak_normalize(segment)

        log_mel = audio_to_log_mel(
            segment,
            sr=sr
        )

        # [64, 309]
        mel_tensor = torch.tensor(
            log_mel,
            dtype=torch.float32
        )

        # Add CNN channel dimension:
        # [1, 64, 309]
        mel_tensor = mel_tensor.unsqueeze(0)

        label_tensor = torch.tensor(
            sample["label"],
            dtype=torch.float32
        )

        return mel_tensor, label_tensor


if __name__ == "__main__":

    train_dataset = DeepAudioDataset(
        split="train"
    )

    val_dataset = DeepAudioDataset(
        split="val"
    )

    print("\n=== DATASET CHECK ===")

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Val samples:",
        len(val_dataset)
    )

    mel, label = train_dataset[0]

    print(
        "Example Mel shape:",
        mel.shape
    )

    print(
        "Example label:",
        label.item()
    )

    print(
        "Mel min:",
        mel.min().item()
    )

    print(
        "Mel max:",
        mel.max().item()
    )