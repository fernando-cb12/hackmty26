from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 8000

SEGMENT_SECONDS = 5

N_FFT = 512
HOP_LENGTH = 128
N_MELS = 64
FMIN = 0
FMAX = 4000


def load_caller_audio(audio_path):
    """
    Load stereo WAV and return only channel 0 (caller).

    Expected dataset format:
    - 8000 Hz
    - stereo
    - channel 0 = caller
    - channel 1 = agent
    """

    audio, sr = sf.read(
        audio_path,
        always_2d=True,
        dtype="float32"
    )

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sr} Hz: {audio_path}"
        )

    if audio.shape[1] < 2:
        raise ValueError(
            f"Expected stereo audio, got "
            f"{audio.shape[1]} channel(s): {audio_path}"
        )

    caller = audio[:, 0]

    return caller, sr


def segment_audio(
    audio,
    sr=SAMPLE_RATE,
    segment_seconds=SEGMENT_SECONDS
):
    """
    Split waveform into fixed-length, non-overlapping segments.

    The last segment is dropped if it is shorter than
    segment_seconds.
    """

    segment_samples = int(sr * segment_seconds)

    segments = []

    for start in range(
        0,
        len(audio) - segment_samples + 1,
        segment_samples
    ):
        end = start + segment_samples

        segments.append(audio[start:end])

    return segments


def rms_energy(audio):
    """
    Compute Root Mean Square energy.

    This is NOT used as a classification feature.

    It is currently used only to detect whether a segment
    contains very little acoustic activity.
    """

    return float(
        np.sqrt(
            np.mean(np.square(audio))
        )
    )


def select_high_energy_segment(segments):
    """
    Return the segment with the highest RMS energy.

    This is useful during visualization so that we do not
    accidentally compare two mostly silent segments.
    """

    if not segments:
        raise ValueError("No audio segments available.")

    energies = [
        rms_energy(segment)
        for segment in segments
    ]

    best_index = int(np.argmax(energies))

    return (
        segments[best_index],
        best_index,
        energies[best_index]
    )


def audio_to_log_mel(
    audio,
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
):
    """
    Convert waveform into a Log-Mel spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        center=False
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel.astype(np.float32)


if __name__ == "__main__":

    wav_files = sorted(
        Path("audio").glob("*.wav")
    )

    if not wav_files:
        raise FileNotFoundError(
            "No WAV files found inside audio/"
        )

    audio_path = wav_files[0]

    caller, sr = load_caller_audio(audio_path)

    print("\n=== AUDIO ===")
    print("File:", audio_path)
    print("Sample rate:", sr)
    print("Caller shape:", caller.shape)
    print(
        "Duration:",
        round(len(caller) / sr, 2),
        "seconds"
    )
    print(
        "Min amplitude:",
        float(np.min(caller))
    )
    print(
        "Max amplitude:",
        float(np.max(caller))
    )

    segments = segment_audio(caller)

    print("\n=== SEGMENTS ===")
    print(
        "Number of segments:",
        len(segments)
    )

    if segments:
        print(
            "Samples per segment:",
            len(segments[0])
        )

        print(
            "Seconds per segment:",
            len(segments[0]) / sr
        )

    print("\n=== RMS ENERGY ===")

    for i, segment in enumerate(segments):

        energy = rms_energy(segment)

        print(
            f"Segment {i:02d}: "
            f"RMS = {energy:.6f}"
        )

    best_segment, best_index, best_energy = (
        select_high_energy_segment(segments)
    )

    print("\n=== SELECTED SEGMENT ===")
    print(
        "Highest-energy segment:",
        best_index
    )
    print(
        "RMS:",
        round(best_energy, 6)
    )

    log_mel = audio_to_log_mel(
        best_segment
    )

    print("\n=== LOG-MEL ===")
    print(
        "Shape:",
        log_mel.shape
    )
    print(
        "Min:",
        float(log_mel.min())
    )
    print(
        "Max:",
        float(log_mel.max())
    )