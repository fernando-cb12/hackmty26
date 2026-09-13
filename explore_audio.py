from pathlib import Path
import soundfile as sf
from collections import Counter
import statistics

wav_files = list(Path("audio").rglob("*.wav"))

sample_rates = Counter()
channel_counts = Counter()
durations = []
errors = []

for path in wav_files:
    try:
        info = sf.info(path)

        sample_rates[info.samplerate] += 1
        channel_counts[info.channels] += 1
        durations.append(info.duration)

    except Exception as e:
        errors.append((path, str(e)))

print("\n=== DATASET AUDIO REPORT ===")
print("Total WAV:", len(wav_files))
print("Sample rates:", sample_rates)
print("Channels:", channel_counts)

if durations:
    print(f"Min duration: {min(durations):.2f} s")
    print(f"Max duration: {max(durations):.2f} s")
    print(f"Average duration: {sum(durations) / len(durations):.2f} s")
    print(f"Median duration: {statistics.median(durations):.2f} s")
    print(f"Total audio: {sum(durations) / 3600:.2f} hours")
print("Errors:", len(errors))

if errors:
    print("\nFiles with errors:")
    for path, error in errors:
        print(path, "->", error)