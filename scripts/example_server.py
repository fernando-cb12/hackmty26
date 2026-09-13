"""Production /detect service for the Altur challenge.

Default live configuration uses the two fast, validated candidates:
conversation behavior and acoustic features. Deep audio and semantic analysis
remain benchmarkable candidates, but are intentionally gated out of the live
endpoint until validation latency/error-overlap evidence supports adding them.

Run:
    python scripts/example_server.py --port 8000
    python scripts/check_endpoint.py --url http://localhost:8000/detect --split val --n 0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acoustic.predict import load_model as load_acoustic_model
from acoustic.predict import predict_wav_bytes as predict_acoustic_bytes
from team3_behavior.predict import load_model as load_behavior_model
from team3_behavior.predict import predict_from_turns
from team3_behavior.vad import detect_turns_from_wav_bytes


DEFAULT_WEIGHTS = {"behavior": 0.5, "acoustic": 0.5}


def validate_wav_bytes(wav_bytes: bytes, expected_rate: int = 8000, expected_channels: int = 2) -> None:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
    if channels != expected_channels:
        raise ValueError(f"expected {expected_channels} channels, got {channels}")
    if sample_rate != expected_rate:
        raise ValueError(f"expected {expected_rate} Hz, got {sample_rate} Hz")
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM, got {sample_width * 8}-bit samples")
    if frames <= 0:
        raise ValueError("empty WAV payload")


def parse_request(body: bytes) -> bytes:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    if payload.get("sample_rate", 8000) != 8000:
        raise ValueError(f"sample_rate must be 8000, got {payload.get('sample_rate')}")
    if payload.get("channels", 2) != 2:
        raise ValueError(f"channels must be 2, got {payload.get('channels')}")
    audio_base64 = payload.get("audio_base64")
    if not isinstance(audio_base64, str):
        raise ValueError("audio_base64 must be a base64 string")
    try:
        wav_bytes = base64.b64decode(audio_base64, validate=True)
    except Exception as error:
        raise ValueError("audio_base64 is not valid base64") from error
    validate_wav_bytes(wav_bytes)
    return wav_bytes


def probability_to_verdict(probability: float, threshold: float = 0.5) -> dict[str, Any]:
    is_synthetic = probability >= threshold
    confidence = probability if is_synthetic else 1.0 - probability
    return {"is_synthetic": bool(is_synthetic), "confidence": round(float(confidence), 6)}


class Detector:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or DEFAULT_WEIGHTS
        load_behavior_model()
        load_acoustic_model()

    def detect(self, wav_bytes: bytes, include_diagnostics: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        turns, vad_diagnostics = detect_turns_from_wav_bytes(wav_bytes)
        behavior = predict_from_turns(turns, vad_diagnostics, started=started)
        acoustic = predict_acoustic_bytes(wav_bytes, turns=turns)

        behavior_probability = float(behavior["probability_synthetic"])
        acoustic_probability = float(acoustic["probability_synthetic"])
        total_weight = sum(self.weights.values())
        probability = (
            behavior_probability * self.weights["behavior"]
            + acoustic_probability * self.weights["acoustic"]
        ) / total_weight
        response = probability_to_verdict(probability)

        if include_diagnostics:
            response["probability_synthetic"] = round(probability, 6)
            response["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            response["models"] = {
                "behavior": {
                    "probability_synthetic": behavior_probability,
                    "confidence": behavior["confidence"],
                },
                "acoustic": {
                    "probability_synthetic": acoustic_probability,
                    "confidence": acoustic["confidence"],
                    "elapsed_ms": acoustic.get("elapsed_ms"),
                },
            }
        return response


DETECTOR: Detector | None = None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/detect":
            return self._reply(404, {"error": "use POST /detect"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            wav_bytes = parse_request(self.rfile.read(length))
            result = DETECTOR.detect(wav_bytes) if DETECTOR else {"error": "detector not initialized"}
        except Exception as error:
            return self._reply(400, {"error": str(error)})
        self._reply(200, result)

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} {fmt % args}")


def main() -> None:
    global DETECTOR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    DETECTOR = Detector()
    print(f"listening on http://{args.host}:{args.port}/detect")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
