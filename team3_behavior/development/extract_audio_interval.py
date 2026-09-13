"""Extract a short caller, agent, or stereo WAV interval for listening."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--channel", choices=["caller", "agent", "stereo"], default="caller")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.end <= args.start:
        raise SystemExit("--end must be greater than --start, and both must be non-negative")

    with wave.open(str(args.wav), "rb") as source:
        if source.getnchannels() != 2 or source.getsampwidth() != 2:
            raise SystemExit("expected a stereo 16-bit PCM WAV")
        sample_rate = source.getframerate()
        samples = np.frombuffer(
            source.readframes(source.getnframes()), dtype="<i2"
        ).reshape(-1, 2)

    start_sample = min(len(samples), round(args.start * sample_rate))
    end_sample = min(len(samples), round(args.end * sample_rate))
    selected = samples[start_sample:end_sample]
    if args.channel == "caller":
        selected = selected[:, 0]
        channels = 1
    elif args.channel == "agent":
        selected = selected[:, 1]
        channels = 1
    else:
        channels = 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(selected.astype("<i2", copy=False).tobytes())
    print(
        f"wrote {args.channel} {args.start:.2f}-{args.end:.2f}s "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
