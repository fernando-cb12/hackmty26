"""Inspect a transcript and its semantic features without loading a classifier."""

from __future__ import annotations

import argparse
import json

try:  # Supports both `python team4_semantic/analyze.py` and package imports.
    from .feature_extraction import extract_features
    from .transcribe import transcribe_call
except ImportError:  # pragma: no cover - direct-script compatibility
    from feature_extraction import extract_features
    from transcribe import transcribe_call


def main() -> None:
    parser = argparse.ArgumentParser(description="Show paired Spanish transcripts and semantic features.")
    parser.add_argument("audio")
    parser.add_argument("--turns")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    utterances = transcribe_call(args.audio, args.turns, args.cache_dir, device=args.device)
    print(json.dumps({"utterances": utterances, "features": extract_features(utterances)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
