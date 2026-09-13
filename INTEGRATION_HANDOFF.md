# Integration Handoff

Use this on the stronger device. This repo now has a live `/detect` service that
defaults to behavior + acoustic fusion, plus an offline runner for comparing
the gated deep-audio and semantic candidates.

## 1. Prepare the environment

From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r acoustic/requirements.txt
.venv/bin/python -m pip install -r team3_behavior/requirements.txt
```

Make sure `audio/`, `manifest.csv`, and `turns/` are present in the repo root.
The live endpoint does not depend on `turns/` for hidden calls; it derives turns
from runtime VAD.

## 2. Smoke-test the default endpoint

Terminal A:

```bash
.venv/bin/python scripts/example_server.py --port 8000
```

Terminal B:

```bash
.venv/bin/python scripts/check_endpoint.py \
  --url http://localhost:8000/detect \
  --split val \
  --n 10 \
  --out reports/endpoint_smoke.json
```

If that passes, run the full validation split:

```bash
.venv/bin/python scripts/check_endpoint.py \
  --url http://localhost:8000/detect \
  --split val \
  --n 0 \
  --out reports/endpoint_val.json
```

Record balanced accuracy, AUC, Brier score, mean latency, and max latency.

## 3. Run candidate comparison

Default, fast candidates:

```bash
.venv/bin/python scripts/ensemble_experiment.py \
  --split val \
  --n 0 \
  --out reports/ensemble_val.json
```

Optional heavier candidates:

```bash
.venv/bin/python -m pip install torch
.venv/bin/python scripts/ensemble_experiment.py \
  --split val \
  --n 0 \
  --include-deep \
  --out reports/ensemble_deep_val.json
```

Semantic analysis needs the semantic requirements and trained model artifact:

```bash
.venv/bin/python -m pip install -r team4_semantic/requirements.txt
.venv/bin/python scripts/ensemble_experiment.py \
  --split val \
  --n 0 \
  --include-semantic \
  --out reports/ensemble_semantic_val.json
```

Only include deep or semantic in the live endpoint if it improves held-out
balanced accuracy or fixes distinct errors while keeping P95 latency under 10s.

## 4. Final judging checklist

- Keep the server process running and reachable at the URL you give judges.
- Re-run `scripts/check_endpoint.py --split val --n 0` shortly before judging.
- Confirm every response has boolean `is_synthetic` and numeric `confidence`.
- Keep `reports/endpoint_val.json` and `reports/ensemble_val.json` for demo
  evidence: metrics, confusion matrix, false positives/negatives, and error
  overlap.
- Pitch the system as four independent signals, with the live endpoint using
  the validated fast subset: acoustic traits + conversational behavior.
