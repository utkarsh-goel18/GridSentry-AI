import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from backend.inference import engine

print("\nLoading trained model...")
engine.load()

print(f"Model: {engine.model_metadata['model_name']}")
print(f"Device: {engine.device}")
print(f"Classes: {len(engine.classes)}")

samples = engine.get_available_samples(limit=20)

if not samples:
    raise RuntimeError("No Dataset 1 samples found.")

print(f"\nTesting {len(samples)} representative samples...\n")

passed = 0
failed = 0

for i, sample in enumerate(samples, 1):
    sample_id = sample["sample_id"]
    actual = sample["fault_class"]

    try:
        result = engine.predict(sample_id=sample_id)

        predicted = result["predicted_fault"]
        confidence = result["confidence"]
        probabilities = result["class_probabilities"]
        probability_sum = sum(probabilities.values())

        valid = (
            predicted in engine.classes
            and np.isfinite(confidence)
            and abs(probability_sum - 1.0) < 1e-4
            and all(np.isfinite(v) for v in probabilities.values())
        )

        correct = predicted == actual

        if valid:
            passed += 1
        else:
            failed += 1

        print(f"[{i}/{len(samples)}] {sample_id}")
        print(f"  Actual:       {actual}")
        print(f"  Predicted:    {predicted}")
        print(f"  Confidence:   {confidence:.4f}")
        print(f"  Probability sum: {probability_sum:.6f}")
        print(f"  Correct:      {correct}")
        print(f"  Valid output: {valid}")
        print(f"  Inference:    {result['inference_time_ms']:.2f} ms\n")

    except Exception as exc:
        failed += 1
        print(f"[{i}/{len(samples)}] FAILED: {sample_id}")
        print(f"  Error: {exc}\n")

print("=" * 50)
print("INFERENCE SMOKE TEST SUMMARY")
print("=" * 50)
print(f"Total samples: {len(samples)}")
print(f"Valid predictions: {passed}")
print(f"Failed predictions: {failed}")
print("Note: This is a pipeline smoke test, not a test-set accuracy evaluation.")
