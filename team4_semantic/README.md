# Team 4: Semantic / Response Analysis

This component classifies the caller from **what is said**, not from caller acoustics. It transcribes Spanish channel-0 and channel-1 turns, pairs each caller response with the preceding agent prompt, and computes transparent conversational-content features.

## Pipeline

1. `transcribe.py` reads stereo 8 kHz PCM WAV audio, uses the supplied turn boundaries, merges adjacent fragments, and transcribes each channel with offline `faster-whisper` (`small`, Spanish). Transcript caches are invalidated whenever the audio, turns file, or Whisper model name changes.
2. `feature_extraction.py` measures lexical diversity, fillers, uncertainty/refusal/clarification language, repeated responses, response relevance, detail density, unsupported details, and numeric inconsistency. Agent text is context only. Audio quality and STT confidence are deliberately excluded.
3. `train.py` compares a calibrated explainable-feature logistic model, a calibrated character-TF-IDF caller-text model, and their average. It selects a fusion only when 5-fold train out-of-fold ROC-AUC improves by at least 0.005. The selected threshold maximizes train OOF F1.

The embedding feature uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. If it cannot be loaded, feature extraction falls back to normalized lexical vectors; training reports should record that environment limitation.

## Setup

```powershell
.\venv\Scripts\python.exe -m pip install -r team4_semantic\requirements.txt
```

The first transcription downloads Whisper weights; the first semantic-embedding calculation downloads multilingual embedding weights. Both are cached by their libraries and subsequent runs remain local.

## Train and evaluate

Run from the repository root:

```powershell
.\venv\Scripts\python.exe team4_semantic\train.py --device auto
```

It writes the fitted pipeline to `team4_semantic/model/semantic_model.joblib`, metrics to `metrics.json`, and per-call validation predictions to `validation_predictions.csv`. Metrics include accuracy, precision, recall, F1, ROC-AUC, Brier calibration score, confusion matrix, and total experiment duration. Audio/transcript cache files are excluded from Git.

## Inference

```powershell
.\venv\Scripts\python.exe team4_semantic\predict.py audio\call_0181ce113ebe.wav --turns turns\call_0181ce113ebe.json
```

Python integration:

```python
from team4_semantic import predict

result = predict("audio/call_0181ce113ebe.wav", "turns/call_0181ce113ebe.json")
# {"is_synthetic": True, "probability": 0.68,
#  "reasoning_features": {...}, "inference_time_s": 1.23}
```

For the final service, map `probability` to its `confidence` field. The semantic score should remain an independently validated input to final model fusion.

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s team4_semantic\tests -v
```

The tests cover channel-safe turn merging/pairing, explainable Spanish feature extraction, deterministic schema order, and empty-caller handling. Run training to measure dataset metrics and inspect false positives/negatives from `validation_predictions.csv`.

## Windows CUDA troubleshooting

If `--device auto` reports a missing CUDA DLL such as `cublas64_12.dll`, the pipeline now retries the current utterance on CPU automatically. To skip the failed CUDA attempt entirely, train or predict with:

```powershell
.\venv\Scripts\python.exe team4_semantic\train.py --device cpu
```

The Hugging Face Xet and Windows symlink messages are cache-performance warnings only; they do not affect transcription correctness.
