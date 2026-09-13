#!/usr/bin/env python3
"""Build an ignored Postman environment containing one complete WAV as base64."""

from __future__ import annotations

import argparse
import base64
import json
import uuid
import wave
from pathlib import Path


DEFAULT_OUTPUT = Path("postman/HackMTY26.generated.postman_environment.json")


def validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
    if channels != 2:
        raise ValueError(f"expected a stereo WAV, got {channels} channel(s)")
    if sample_rate != 8000:
        raise ValueError(f"expected 8000 Hz, got {sample_rate} Hz")
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM, got {sample_width * 8}-bit samples")
    if frames <= 0:
        raise ValueError("WAV contains no audio frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="complete challenge WAV to send")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="server origin without /detect or a trailing slash",
    )
    parser.add_argument("--call-id", help="defaults to the WAV filename without .wav")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    validate_wav(args.wav)
    encoded_audio = base64.b64encode(args.wav.read_bytes()).decode("ascii")
    environment = {
        "id": str(uuid.uuid4()),
        "name": "HackMTY 2026 - Generated Local Test",
        "values": [
            {
                "key": "base_url",
                "value": args.base_url.rstrip("/"),
                "type": "default",
                "enabled": True,
            },
            {
                "key": "call_id",
                "value": args.call_id or args.wav.stem,
                "type": "default",
                "enabled": True,
            },
            {
                "key": "audio_base64",
                "value": encoded_audio,
                "type": "secret",
                "enabled": True,
            },
        ],
        "_postman_variable_scope": "environment",
        "_postman_exported_at": "2026-09-13T00:00:00.000Z",
        "_postman_exported_using": "HackMTY build_environment.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    size_mib = args.output.stat().st_size / (1024 * 1024)
    print(f"wrote {args.output} ({size_mib:.2f} MiB)")
    print("Import this environment into Postman and select it before running the collection.")


if __name__ == "__main__":
    main()
