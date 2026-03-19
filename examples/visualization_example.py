"""
Visualization Example for PyImgAno.

This script demonstrates visualization capabilities for anomaly detection results.
"""

from pathlib import Path

import numpy as np

from pyimgano import models, visualization

JPG_GLOB = "*.jpg"



def main():
    """Run visualization example."""
    print("=" * 60)
    print("PyImgAno Visualization Example")
    print("=" * 60 + "\n")

    # Setup paths
    train_dir = Path("data/train/normal")
    test_dir = Path("data/test")

    if not train_dir.exists():
        print("⚠️  Data directory not found.")
        return

    # Collect images
    train_images = [str(p) for p in train_dir.glob(JPG_GLOB)]
    train_images += [str(p) for p in train_dir.glob("*.png")]

    if len(train_images) == 0:
        print("⚠️  No training images found!")
        return

    print(f"Using {len(train_images)} training images\n")

    # Create and train detector (use a model that supports anomaly maps)
    print("Training PatchCore detector...")
    detector = models.create_model("vision_patchcore", coreset_sampling_ratio=0.1, device="cpu")

    detector.fit(train_images[: min(20, len(train_images))])  # Use subset for speed
    print("✓ Training complete\n")

    # Find a test image
    test_images = list((test_dir / "anomaly").glob(JPG_GLOB))
    if not test_images:
        test_images = list((test_dir / "normal").glob(JPG_GLOB))

    if not test_images:
        print("⚠️  No test images found!")
        return

    test_image = str(test_images[0])
    print(f"Analyzing: {test_image}\n")

    # Generate anomaly map
    print("Generating anomaly map...")
    anomaly_map = detector.get_anomaly_map(test_image)
    print("✓ Anomaly map generated\n")

    # Visualize
    try:
        print("Creating visualization...")
        visualization.plot_anomaly_map(
            test_image,
            anomaly_map,
            alpha=0.5,
            save_path="anomaly_visualization.png",
            show=False,  # Set to True to display
        )
        print("✓ Visualization saved to: anomaly_visualization.png")

        # Also save simple overlay
        visualization.save_anomaly_overlay(
            test_image, anomaly_map, "anomaly_overlay.jpg", alpha=0.5
        )
        print("✓ Overlay saved to: anomaly_overlay.jpg")

    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        print("Note: matplotlib may not be installed or display may not be available")

    # Compare multiple detectors
    if len(train_images) >= 10:
        print("\nComparing multiple detectors...")

        # Train additional detectors
        detectors = {
            "PatchCore": detector,  # Already trained
            "ECOD": models.create_model("vision_ecod", contamination=0.1),
            "COPOD": models.create_model("vision_copod", contamination=0.1),
        }

        # Train the rest
        for name, det in detectors.items():
            if name != "PatchCore":
                print(f"Training {name}...")
                det.fit(train_images[: min(20, len(train_images))])

        # Collect test images and labels
        test_normal = list((test_dir / "normal").glob(JPG_GLOB))
        test_anomaly = list((test_dir / "anomaly").glob(JPG_GLOB))

        if test_normal and test_anomaly:
            test_paths = [str(p) for p in test_normal + test_anomaly]
            test_labels = np.array([0] * len(test_normal) + [1] * len(test_anomaly))

            # Get scores from all detectors
            scores_dict = {}
            for name, det in detectors.items():
                print(f"Running inference with {name}...")
                scores_dict[name] = det.predict(test_paths)

            # Compare
            print("\nCreating comparison plot...")
            try:
                visualization.compare_detectors(
                    test_labels, scores_dict, save_path="detector_comparison.png", show=False
                )
                print("✓ Comparison saved to: detector_comparison.png")
            except Exception as e:
                print(f"⚠️  Comparison plot failed: {e}")

    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
