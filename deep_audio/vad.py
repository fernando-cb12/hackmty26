import numpy as np


FRAME_MS = 30
HOP_MS = 10


def frame_audio(audio, sr=8000, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    frame_size = int(sr * frame_ms / 1000)
    hop_size = int(sr * hop_ms / 1000)

    frames = []

    for start in range(0, len(audio) - frame_size + 1, hop_size):
        end = start + frame_size
        frames.append(audio[start:end])

    return frames


def frame_rms(frame):
    return float(np.sqrt(np.mean(frame ** 2)))


def speech_mask(audio, sr=8000, threshold_db=20):
    """
    Estimate speech-active frames using an adaptive energy threshold.

    threshold_db:
        how far above the estimated noise floor a frame must be
        to count as active.
    """

    frames = frame_audio(audio, sr)

    if not frames:
        return np.array([], dtype=bool)

    rms_values = np.array(
        [frame_rms(frame) for frame in frames],
        dtype=np.float32
    )

    eps = 1e-10

    rms_db = 20 * np.log10(rms_values + eps)

    # Estimate noise floor using low-energy frames.
    noise_floor = np.percentile(rms_db, 20)

    threshold = noise_floor + threshold_db

    mask = rms_db > threshold

    return mask


def speech_ratio(audio, sr=8000, threshold_db=20):
    mask = speech_mask(
        audio,
        sr=sr,
        threshold_db=threshold_db
    )

    if len(mask) == 0:
        return 0.0

    return float(np.mean(mask))


if __name__ == "__main__":
    from pathlib import Path

    from deep_audio.preprocessing import (
        load_caller_audio,
        segment_audio,
    )

    wav_files = sorted(
        Path("audio").glob("*.wav")
    )

    audio_path = wav_files[0]

    caller, sr = load_caller_audio(audio_path)
    segments = segment_audio(caller)

    print("File:", audio_path)

    for i, segment in enumerate(segments):
        ratio = speech_ratio(segment, sr)

        print(
            f"Segment {i:02d}: "
            f"speech_ratio = {ratio:.3f}"
        )