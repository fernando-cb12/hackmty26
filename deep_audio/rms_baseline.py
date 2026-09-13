import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from deep_audio.preprocessing import (
    load_caller_audio,
    segment_audio,
    rms_energy,
)


MANIFEST_PATH = Path("manifest.csv")
AUDIO_DIR = Path("audio")


def extract_call_features(call_id):
    """
    Extract simple RMS statistics for one complete call.

    IMPORTANT:
    These features are only used as a sanity baseline.
    They are NOT part of the final deep audio model.
    """

    audio_path = AUDIO_DIR / f"{call_id}.wav"

    caller, _ = load_caller_audio(audio_path)
    segments = segment_audio(caller)

    rms_values = np.array(
        [rms_energy(segment) for segment in segments],
        dtype=np.float32
    )

    return [
        float(np.mean(rms_values)),
        float(np.median(rms_values)),
        float(np.percentile(rms_values, 90)),
    ]


X_train = []
y_train = []

X_val = []
y_val = []


with open(MANIFEST_PATH, newline="") as f:
    reader = csv.DictReader(f)

    rows = list(reader)


for i, row in enumerate(rows, start=1):

    call_id = row["anon_id"]
    label = row["label"]
    split = row["split"]

    features = extract_call_features(call_id)

    # synthetic = 1
    # human = 0
    target = 1 if label == "synthetic" else 0

    if split == "train":
        X_train.append(features)
        y_train.append(target)

    elif split == "val":
        X_val.append(features)
        y_val.append(target)

    print(
        f"[{i:03d}/{len(rows)}] "
        f"{call_id} -> {label}"
    )


X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = np.array(y_val)


print("\n=== DATA ===")

print("Train calls:", len(X_train))
print("Validation calls:", len(X_val))

print("Features:")
print("  mean RMS")
print("  median RMS")
print("  P90 RMS")


model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)


probabilities = model.predict_proba(
    X_val
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


accuracy = accuracy_score(
    y_val,
    predictions
)

precision = precision_score(
    y_val,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_val,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_val,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_val,
    probabilities
)


print("\n=== RMS-ONLY BASELINE ===")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


print("\n=== MODEL COEFFICIENTS ===")

feature_names = [
    "mean_rms",
    "median_rms",
    "p90_rms"
]

for name, coefficient in zip(
    feature_names,
    model.coef_[0]
):
    print(
        f"{name:12s}: "
        f"{coefficient:.4f}"
    )