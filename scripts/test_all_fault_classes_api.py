import csv
import json
import urllib.request
import zipfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.loader import MatlabSignalLoader
from ml.config import (
    DATASET_1_ZIP,
    D1_FAULT_INDEX,
    PRE_FAULT_SAMPLES,
    STANDARD_CLASSES,
)

loader = MatlabSignalLoader()
selected = {}

print("Scanning Dataset 1 for representative recordings...")

with zipfile.ZipFile(DATASET_1_ZIP, "r") as archive:
    for filename in sorted(archive.namelist()):
        if not filename.endswith(".mat") or "Classifier_LOC3" not in filename:
            continue

        metadata = loader.parse_filename_metadata(filename)
        label = metadata.get("standard_fault")

        if label in STANDARD_CLASSES and label not in selected:
            selected[label] = Path(filename).name

missing = set(STANDARD_CLASSES) - set(selected)
if missing:
    raise RuntimeError(f"Missing recordings for classes: {sorted(missing)}")

results = []

print(f"\nFound recordings for all {len(STANDARD_CLASSES)} classes.")
print("Sending waveforms to http://127.0.0.1:8000/predict\n")

for expected in STANDARD_CLASSES:
    sample_id = selected[expected]
    waveform, metadata = loader.load_dataset_1_sample(sample_id)

    payload = {
        "waveform": waveform.tolist(),
        "fault_index": D1_FAULT_INDEX,
        "pre_fault_samples": PRE_FAULT_SAMPLES,
    }

    request = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))

        predicted = result["predicted_fault"]
        confidence = result["confidence"]
        latency = result["inference_time_ms"]
        passed = predicted == expected

        results.append({
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "latency_ms": latency,
            "status": "PASS" if passed else "FAIL",
            "sample_id": sample_id,
        })

        print(
            f"{expected:4} -> {predicted:4} | "
            f"Confidence: {confidence:8.4f} | "
            f"Latency: {latency:7.2f} ms | "
            f"{'PASS' if passed else 'FAIL'}"
        )

    except Exception as error:
        print(f"{expected:4} | ERROR: {error}")
        raise

output = Path("artifacts/ten_class_api_results.csv")
output.parent.mkdir(exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

correct = sum(row["status"] == "PASS" for row in results)
average_latency = sum(row["latency_ms"] for row in results) / len(results)

print("\n" + "=" * 65)
print(f"CORRECT PREDICTIONS : {correct}/{len(results)}")
print(f"CHECK ACCURACY      : {correct / len(results) * 100:.2f}%")
print(f"AVERAGE LATENCY     : {average_latency:.2f} ms")
print(f"RESULTS CSV         : {output.resolve()}")
print("=" * 65)

if correct != len(results):
    print("Some representative recordings were misclassified.")
else:
    print("All ten representative API integration checks passed.")
