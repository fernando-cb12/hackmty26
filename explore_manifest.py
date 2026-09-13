import csv
from pathlib import Path
from collections import Counter

rows = []

with open("manifest.csv", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("=== MANIFEST REPORT ===")
print("Total rows:", len(rows))

print("\nLabels:")
print(Counter(row["label"] for row in rows))

print("\nSplits:")
print(Counter(row["split"] for row in rows))

print("\nLabel x Split:")
combinations = Counter(
    (row["split"], row["label"])
    for row in rows
)

for key, count in sorted(combinations.items()):
    print(key, count)

# Check that every manifest entry has a WAV
missing = []

for row in rows:
    path = Path("audio") / f'{row["anon_id"]}.wav'

    if not path.exists():
        missing.append(path)

print("\nMissing WAV files:", len(missing))

for path in missing[:20]:
    print(" -", path)