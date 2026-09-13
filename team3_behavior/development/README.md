# Desarrollo del Modelo 3

Esta carpeta contiene todo lo que no se necesita durante una predicción:
entrenamiento, evaluación, comparación del VAD, diagnósticos y visualizaciones.
Los comandos se ejecutan desde la raíz del repositorio para reutilizar
`audio/`, `manifest.csv` y `turns/`.

## Estructura

```text
development/
├── train_v2.py
├── evaluate.py
├── compare_vad.py
├── visualize_errors.py
├── diagnose_energy.py
├── visualize_energy.py
├── extract_audio_interval.py
├── analyze.py
├── train_v1.py
└── artifacts/
    ├── features/   # tablas derivadas de los datos compartidos
    ├── models/     # métricas, predicciones y modelos anteriores
    └── reports/    # reportes visuales y diagnósticos
```

## Regenerar las características V2

```bash
python3 -m team3_behavior.feature_extraction \
  --manifest manifest.csv \
  --audio-dir audio \
  --feature-version v2 \
  --output team3_behavior/development/artifacts/features/features_energy_vad_v2.csv
```

## Reentrenar

```bash
python3 -m team3_behavior.development.train_v2 \
  --features team3_behavior/development/artifacts/features/features_energy_vad_v2.csv \
  --model-out team3_behavior/model/behavior_v2.joblib \
  --report-out team3_behavior/development/artifacts/models/behavior_v2_report.json \
  --predictions-out team3_behavior/development/artifacts/models/val_predictions_v2.csv
```

La selección compara regresión logística, Random Forest e
HistGradientBoosting mediante validación cruzada de cinco partes dentro de
`train`. El split `val` sólo se usa al final para reportar el resultado.

## Evaluar el recorrido completo

```bash
python3 -m team3_behavior.development.evaluate \
  --manifest manifest.csv \
  --audio-dir audio \
  --split val
```

## Comparar el VAD con los turnos oficiales

```bash
python3 -m team3_behavior.development.compare_vad \
  --manifest manifest.csv \
  --audio-dir audio \
  --turns-dir turns \
  --split all
```

## Visualizar errores

```bash
python3 -m team3_behavior.development.visualize_errors \
  --predictions team3_behavior/development/artifacts/models/val_predictions_v2.csv \
  --audio-dir audio \
  --turns-dir turns \
  --output team3_behavior/development/artifacts/reports/v2_error_timelines.html
```

## Diagnosticar una llamada

```bash
python3 -m team3_behavior.development.diagnose_energy \
  audio/call_b712f501aeb8.wav \
  --official-turns turns/call_b712f501aeb8.json \
  --json-output team3_behavior/development/artifacts/reports/call_b712_energy_diagnosis.json \
  --csv-output team3_behavior/development/artifacts/reports/call_b712_extra_intervals.csv

python3 -m team3_behavior.development.visualize_energy \
  team3_behavior/development/artifacts/reports/call_b712_energy_diagnosis.json \
  --output team3_behavior/development/artifacts/reports/call_b712_energy.svg
```

Los WAV recortados y los reportes de diagnóstico grandes quedan disponibles
localmente, pero `.gitignore` evita que se suban al repositorio.
