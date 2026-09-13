"""Turn-aware Spanish transcription for the semantic detector."""

from __future__ import annotations

import argparse
import json
import os
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class Utterance:
    channel: int
    start: float
    end: float
    text: str = ""
    preceding_agent: str = ""
    response_gap_s: float | None = None


def default_turns_path(audio_path: str | Path) -> Path | None:
    """Find the dataset turns file when the caller did not supply one."""
    audio = Path(audio_path)
    candidate = audio.parent.parent / "turns" / f"{audio.stem}.json"
    return candidate if candidate.exists() else None


def load_turns(turns_path: str | Path | None, duration_s: float) -> list[dict[str, Any]]:
    if turns_path is None:
        # This fallback preserves channel separation when a deployment cannot
        # provide VAD turns. It is less accurate but keeps prediction usable.
        return [
            {"channel": 0, "start": 0.0, "end": duration_s},
            {"channel": 1, "start": 0.0, "end": duration_s},
        ]
    payload = json.loads(Path(turns_path).read_text(encoding="utf-8"))
    turns = payload.get("turns", [])
    valid = []
    for turn in turns:
        channel, start, end = turn.get("channel"), turn.get("start"), turn.get("end")
        if channel in (0, 1) and isinstance(start, (int, float)) and isinstance(end, (int, float)):
            if 0 <= start < end <= duration_s + 0.1:
                valid.append({"channel": channel, "start": float(start), "end": float(end)})
    return sorted(valid, key=lambda item: (item["start"], item["end"]))


def merge_turns(turns: Iterable[dict[str, Any]], max_gap_s: float = 0.75) -> list[Utterance]:
    """Join consecutive same-speaker fragments without crossing another turn."""
    merged: list[Utterance] = []
    for turn in sorted(turns, key=lambda item: (item["start"], item["end"])):
        current = Utterance(turn["channel"], turn["start"], turn["end"])
        if (
            merged
            and merged[-1].channel == current.channel
            and current.start - merged[-1].end <= max_gap_s
        ):
            merged[-1].end = max(merged[-1].end, current.end)
        else:
            merged.append(current)
    return merged


def _read_pcm16_stereo(audio_path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as source:
        if source.getnchannels() != 2 or source.getsampwidth() != 2:
            raise ValueError("Expected a stereo 16-bit PCM WAV file")
        rate = source.getframerate()
        raw = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return raw.reshape(-1, 2).astype(np.float32) / 32768.0, rate


class WhisperTranscriber:
    """Lazy faster-whisper wrapper with CUDA-first, CPU-safe execution."""

    def __init__(self, model_name: str = "small", device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None
        self._active_device: str | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        selected = self.device
        if selected == "auto":
            try:
                import ctranslate2

                selected = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                selected = "cpu"
        compute_type = "int8_float16" if selected == "cuda" else "int8"
        try:
            self._model = WhisperModel(self.model_name, device=selected, compute_type=compute_type)
            self._active_device = selected
        except Exception:
            if selected != "cuda":
                raise
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            self._active_device = "cpu"
        return self._model

    def _switch_to_cpu(self) -> None:
        """Recover from CUDA DLL errors that appear only during model execution."""
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        self._active_device = "cpu"

    def transcribe_audio(self, samples: np.ndarray, sample_rate: int) -> str:
        if samples.size == 0:
            return ""
        # faster-whisper accepts a NumPy array as already-decoded 16 kHz audio.
        # Our challenge WAVs are 8 kHz telephone calls, so passing them directly
        # doubles playback speed and produces corrupt transcripts.
        if sample_rate != 16000:
            target_length = round(len(samples) * 16000 / sample_rate)
            source_positions = np.arange(len(samples), dtype=np.float64)
            target_positions = np.arange(target_length, dtype=np.float64) * sample_rate / 16000
            samples = np.interp(target_positions, source_positions, samples).astype(np.float32)
        def run() -> str:
            model = self._load_model()
            segments, _ = model.transcribe(
                samples,
                language="es",
                task="transcribe",
                vad_filter=False,
                beam_size=3,
                condition_on_previous_text=False,
            )
            # `segments` is lazy: CUDA loader failures happen while iterating.
            return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

        try:
            return run()
        except (RuntimeError, OSError) as error:
            if self._active_device != "cuda":
                raise
            print(f"CUDA transcription unavailable ({error}); retrying on CPU.")
            self._switch_to_cpu()
            return run()


# Training processes hundreds of calls. Keep one model process-resident rather
# than reloading its weights for every WAV file.
_TRANSCRIBERS: dict[tuple[str, str], WhisperTranscriber] = {}


def get_transcriber(model_name: str, device: str) -> WhisperTranscriber:
    key = (model_name, device)
    if key not in _TRANSCRIBERS:
        _TRANSCRIBERS[key] = WhisperTranscriber(model_name=model_name, device=device)
    return _TRANSCRIBERS[key]


def pair_caller_responses(utterances: list[Utterance]) -> list[Utterance]:
    """Attach only prior agent context to caller utterances; never future context."""
    prior_agent: Utterance | None = None
    for utterance in utterances:
        if utterance.channel == 1:
            prior_agent = utterance
        elif utterance.channel == 0 and prior_agent is not None:
            utterance.preceding_agent = prior_agent.text
            utterance.response_gap_s = max(0.0, utterance.start - prior_agent.end)
    return utterances


def _cache_key(audio: Path, turns: Path | None, model_name: str) -> dict[str, Any]:
    return {
        # Bump whenever transcript preprocessing changes, invalidating stale text.
        "pipeline_version": 2,
        "audio_mtime_ns": audio.stat().st_mtime_ns,
        "turns_mtime_ns": turns.stat().st_mtime_ns if turns and turns.exists() else None,
        "model": model_name,
    }


def transcribe_call(
    audio_path: str | Path,
    turns_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    model_name: str = "small",
    device: str = "auto",
) -> list[dict[str, Any]]:
    """Return transcript utterances, caching a call only when source files match."""
    audio = Path(audio_path)
    turns = Path(turns_path) if turns_path else default_turns_path(audio)
    cache = Path(cache_dir) if cache_dir else Path(__file__).parent / "cache"
    cache_file = cache / f"{audio.stem}.json"
    key = _cache_key(audio, turns, model_name)
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("cache_key") == key:
            return cached["utterances"]

    waveform, rate = _read_pcm16_stereo(audio)
    turns_data = load_turns(turns, len(waveform) / rate)
    utterances = merge_turns(turns_data)
    transcriber = get_transcriber(model_name=model_name, device=device)
    for utterance in utterances:
        left = max(0, int(utterance.start * rate))
        right = min(len(waveform), int(utterance.end * rate))
        # A short pad avoids clipping initial/final phonemes but stays in-channel.
        pad = int(0.12 * rate)
        utterance.text = transcriber.transcribe_audio(
            waveform[max(0, left - pad) : min(len(waveform), right + pad), utterance.channel], rate
        )
    paired = [asdict(item) for item in pair_caller_responses(utterances)]
    cache.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"cache_key": key, "utterances": paired}, ensure_ascii=False), encoding="utf-8")
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a stereo Spanish call using its turn file.")
    parser.add_argument("audio")
    parser.add_argument("--turns")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model", default="small")
    args = parser.parse_args()
    print(json.dumps(transcribe_call(args.audio, args.turns, args.cache_dir, args.model, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
