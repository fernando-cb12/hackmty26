from collections import defaultdict

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from deep_audio.dataset import DeepAudioDataset
from deep_audio.model import DeepAudioCNN


MODEL_PATH = "models/deep_audio_cnn.pt"
THRESHOLD = 0.5


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset = DeepAudioDataset(split="val")

    model = DeepAudioCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    call_probabilities = defaultdict(list)
    call_labels = {}

    print("\nRunning call-level evaluation...")

    with torch.no_grad():

        for index in range(len(dataset)):

            mel, label = dataset[index]

            sample = dataset.samples[index]
            call_id = sample["call_id"]

            mel = mel.unsqueeze(0).to(device)

            logit = model(mel)

            probability = torch.sigmoid(
                logit
            ).item()

            call_probabilities[call_id].append(
                probability
            )

            call_labels[call_id] = int(
                label.item()
            )

    y_true = []
    y_prob = []

    print("\n=== CALL PREDICTIONS ===")

    for call_id in sorted(call_probabilities):

        probabilities = call_probabilities[
            call_id
        ]

        # Baseline aggregation:
        # average segment probability
        call_probability = float(
            np.mean(probabilities)
        )

        label = call_labels[call_id]

        y_true.append(label)
        y_prob.append(call_probability)

        print(
            f"{call_id} | "
            f"segments={len(probabilities):02d} | "
            f"label={label} | "
            f"p_synthetic={call_probability:.4f}"
        )

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_pred = (
        y_prob >= THRESHOLD
    ).astype(int)

    print("\n=== CALL-LEVEL METRICS ===")

    print(
        "Evaluated calls:",
        len(y_true)
    )

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_true, y_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"F1:        "
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{roc_auc_score(y_true, y_prob):.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )


if __name__ == "__main__":
    main()