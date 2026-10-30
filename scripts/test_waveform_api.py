import sys
from pathlib import Path
import json
import urllib.request
import urllib.error
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.loader import MatlabSignalLoader
from ml.config import D1_FAULT_INDEX, PRE_FAULT_SAMPLES

sample_id = "sim_Chesf_500_Classifier_LOC3_AT_150_10_100.000.mat"

print("Loading actual Dataset 1 waveform...")
loader = MatlabSignalLoader()
waveform, metadata = loader.load_dataset_1_sample(sample_id)

print("Waveform shape:", waveform.shape)
print("Actual fault:", metadata["standard_fault"])

payload = {
    "waveform": waveform.tolist(),
    "fault_index": D1_FAULT_INDEX,
    "pre_fault_samples": PRE_FAULT_SAMPLES
}

data = json.dumps(payload).encode("utf-8")

request = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

print("\nSending raw waveform to FastAPI...")

try:
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
        print("HTTP Status:", response.status)
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))

        assert result["input_type"] == "direct_waveform"
        assert result["predicted_fault"] in [
            "AG", "BG", "CG", "AB", "BC",
            "CA", "ABG", "BCG", "CAG", "ABC"
        ]

        print("\nRAW WAVEFORM API TEST PASSED")

except urllib.error.HTTPError as error:
    print("HTTP Error:", error.code)
    print(error.read().decode("utf-8"))
    raise

except Exception as error:
    print("Test failed:", error)
    raise
