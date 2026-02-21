"""
Quick Start Example for PyImgAno.

This script demonstrates basic usage of PyImgAno for anomaly detection.
"""

import os
from pathlib import Path

import numpy as np

from pyimgano import evaluate_detector, models


def main():
    """Run quick start example."""
    print("=" * 60)
    print("PyImgAno Quick Start Example")
    print("=" * 60 + "\n")

    # Note: Replace these with your actual image paths
    print("Setting up paths...")

    # Example paths - adjust to your dataset
    train_dir = Path("data/train/normal")  # Directory with normal training images
    test_normal_dir = Path("data/test/normal")  # Normal test images
    test_anomaly_dir = Path("data/test/anomaly")  # Anomalous test images

    # Check if directories exist
    if not train_dir.exists():
        print(f"⚠️  Training directory not found: {train_dir}")
        print("\nTo run this example:")
        print("1. Create the directory structure:")
        print("   data/train/normal/")
        print("   data/test/normal/")
        print("   data/test/anomaly/")
        print("2. Add your images to these directories")
        print("3. Run this script again")
        return

    # Collect image paths
    train_images = [str(p) for p in train_dir.glob("*.jpg")]
    train_images += [str(p) for p in train_dir.glob("*.png")]

    test_normal = [str(p) for p in test_normal_dir.glob("*.jpg")]
    test_normal += [str(p) for p in test_normal_dir.glob("*.png")]

    test_anomaly = [str(p) for p in test_anomaly_dir.glob("*.jpg")]
    test_anomaly += [str(p) for p in test_anomaly_dir.glob("*.png")]

    test_images = test_normal + test_anomaly
    test_labels = np.array([0] * len(test_normal) + [1] * len(test_anomaly))

    print(f"Found {len(train_images)} training images")
    print(f"Found {len(test_normal)} normal test images")
    print(f"Found {len(test_anomaly)} anomaly test images\n")

    if len(train_images) == 0:
        print("⚠️  No training images found!")
        return

    # Create detector
    print("Creating ECOD detector...")
    detector = models.create_model(
        'vision_ecod',
        contamination=0.1,
        n_jobs=-1
    )

    # Train
    print("Training on normal images...")
    detector.fit(train_images)
    print("✓ Training complete\n")

    # Predict
    print("Running inference on test images...")
    scores = detector.decision_function(test_images)
    predictions = detector.predict(test_images)  # 0=normal, 1=anomaly
    print("✓ Inference complete\n")

    # Evaluate
    if len(test_labels) > 0 and len(np.unique(test_labels)) == 2:
        print("Evaluating performance...")
        results = evaluate_detector(test_labels, scores)

        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        print(f"AUROC:     {results['auroc']:.4f}")
        print(f"Avg Prec:  {results['average_precision']:.4f}")
        print(f"Threshold: {results['threshold']:.4f}")
        print(f"F1:        {results['metrics']['f1']:.4f}")
        print(f"Precision: {results['metrics']['precision']:.4f}")
        print(f"Recall:    {results['metrics']['recall']:.4f}")
        print(f"Predicted anomalies: {int(predictions.sum())}/{len(predictions)}")
        print("=" * 60)
    else:
        print("⚠️  Not enough test data for evaluation")

    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
