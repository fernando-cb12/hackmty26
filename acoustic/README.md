# Teammate 1: acoustic baseline

This classifier uses caller channel 0 and the supplied caller-turn annotations.
It measures voice properties during speech rather than long conversational
silences. Its output is a synthetic probability for the final ensemble.

## Run

From the repository root, use the existing environment:

```bash
./.venv/bin/python acoustic/build_features.py
./.venv/bin/python acoustic/train.py
./.venv/bin/python acoustic/evaluate.py
```

The default audio location is `acoustic/audio/`; change it with `--audio-dir`
if necessary. Feature extraction writes `acoustic/features.csv`; training writes
`acoustic/model/acoustic_model.joblib`.

For one call:

```bash
./.venv/bin/python acoustic/predict.py acoustic/audio/CALL.wav --turns turns/CALL.json
```

`train.py` compares logistic regression, RBF SVM, and random forest on the
official validation split, chooses the best ROC-AUC model, and saves it.
Preprocessing, scaling, and feature selection remain inside each training
pipeline so validation data never leaks into training.

## Feature set

- RMS energy, zero-crossing rate, spectral centroid/bandwidth/rolloff/flatness
- 20 MFCCs plus first and second deltas
- 16-band log-Mel statistics
- F0/pitch statistics from YIN
- caller turn count, speech duration, and turn-duration variation

Framewise features use mean, standard deviation, 10th percentile, median, and
90th percentile. The classifier then selects the 120 strongest features only
from the training set.
