from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from deep_audio.dataset import DeepAudioDataset
from deep_audio.model import DeepAudioCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "deep_audio_cnn.pt"


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, loader, criterion, device):

    model.eval()

    losses = []

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for mels, labels in loader:

            mels = mels.to(device)
            labels = labels.to(device)

            logits = model(mels)

            loss = criterion(
                logits,
                labels
            )

            probabilities = torch.sigmoid(
                logits
            )

            losses.append(
                loss.item()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    labels = np.array(all_labels)
    probabilities = np.array(
        all_probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "loss": np.mean(losses),

        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        ),

        "auc": roc_auc_score(
            labels,
            probabilities
        ),
    }

    return metrics


# --------------------------------------------------
# Main training loop
# --------------------------------------------------

def main():

    print("Loading datasets...")

    train_dataset = DeepAudioDataset(
        split="train"
    )

    val_dataset = DeepAudioDataset(
        split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    model = DeepAudioCNN().to(
        device
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_auc = -1.0

    print("\nStarting training...\n")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ------------------------------
        # Training
        # ------------------------------

        model.train()

        train_losses = []

        for batch_index, (
            mels,
            labels
        ) in enumerate(train_loader):

            mels = mels.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(mels)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

            if (
                batch_index + 1
            ) % 20 == 0:

                print(
                    f"Epoch {epoch:02d} | "
                    f"Batch "
                    f"{batch_index + 1:03d}"
                    f"/{len(train_loader):03d} | "
                    f"Loss "
                    f"{loss.item():.4f}"
                )

        train_loss = np.mean(
            train_losses
        )

        # ------------------------------
        # Validation
        # ------------------------------

        metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss:   "
            f"{metrics['loss']:.4f}"
        )

        print(
            f"Accuracy:   "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision:  "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall:     "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1:         "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:    "
            f"{metrics['auc']:.4f}"
        )

        # ------------------------------
        # Save best model
        # ------------------------------

        if metrics["auc"] > best_auc:

            best_auc = metrics["auc"]

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"✓ Best model saved "
                f"(AUC={best_auc:.4f})"
            )

        print(
            "-" * 50
        )

    print(
        "\nTraining finished."
    )

    print(
        "Best validation AUC:",
        f"{best_auc:.4f}"
    )

    print(
        "Model saved at:",
        MODEL_PATH
    )


if __name__ == "__main__":
    main()
    