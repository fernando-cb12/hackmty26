"""Semantic, transcript-based synthetic-caller detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def predict(
    audio_path: str | Path,
    turns_path: str | Path | None = None,
    model_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Lazy public wrapper, avoiding an eager import when running the CLI module."""
    from .predict import predict as _predict

    return _predict(audio_path, turns_path, model_path, cache_dir, device)


__all__ = ["predict"]
