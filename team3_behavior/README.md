# Modelo 3: comportamiento conversacional

Este modelo distingue llamadas humanas y sintéticas a partir de la dinámica de
la conversación: pausas, latencias de respuesta, duración y regularidad de los
turnos, interrupciones y solapamientos entre caller y agente.

## Qué se necesita para correrlo

La carpeta principal contiene únicamente la ruta de inferencia:

```text
team3_behavior/
├── model/behavior_v2.joblib   # modelo entrenado
├── feature_extraction.py      # convierte turnos en 63 métricas
├── vad.py                     # detecta voz en ambos canales del WAV
├── predict.py                 # devuelve la predicción
├── requirements.txt
└── development/               # entrenamiento, evaluación y diagnósticos
```

El modelo ya está entrenado. Para predecir no usa `manifest.csv` ni los JSON de
`turns/`: sólo necesita el WAV estéreo completo y el archivo `.joblib`.

Desde la raíz del repositorio:

```bash
python3 -m pip install -r team3_behavior/requirements.txt
python3 -m team3_behavior.predict audio/call_0181ce113ebe.wav
```

La respuesta incluye `is_synthetic` y `confidence`, que son los campos útiles
para el endpoint, además de la probabilidad, las características y un resumen
del VAD para facilitar depuración.

Para integrar los bytes recibidos por `POST /detect`:

```python
from team3_behavior.predict import predict_wav_bytes

result = predict_wav_bytes(wav_bytes)
response = {
    "is_synthetic": result["is_synthetic"],
    "confidence": result["confidence"],
}
```

## Datos compartidos

Los tres recursos del reto viven una sola vez en la raíz del repositorio:

- `audio/` para los WAV locales; está ignorado por Git y no debe subirse.
- `manifest.csv` para etiquetas y splits.
- `turns/` para los turnos oficiales usados sólo al desarrollar y comparar.

Ninguno se copia dentro de `team3_behavior/`. Las instrucciones para regenerar
características, reentrenar y visualizar errores están en
[`development/README.md`](development/README.md).

## Resultado actual

El modelo seleccionado es una regresión logística con 63 características y un
umbral de `0.665`, elegidos usando validación cruzada dentro de `train`. En las
71 llamadas de `val` obtuvo balanced accuracy `0.986`, ROC-AUC `0.997`, un falso
positivo y cero falsos negativos.
