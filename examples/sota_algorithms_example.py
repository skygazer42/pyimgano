"""
Example usage of State-of-the-Art (SOTA) anomaly detection algorithms.

This script demonstrates the latest SOTA algorithms added to PyImgAno:
- CutPaste (CVPR 2021): Self-supervised learning
- WinCLIP (CVPR 2023): Zero-shot CLIP-based detection
- DifferNet (WACV 2023): Learnable difference detection
"""

import cv2
import numpy as np

from pyimgano.models import create_model


def generate_sample_data(n_normal=100, n_anomaly=20, image_size=(256, 256)):
    """Generate synthetic image data for testing.

    Args:
        n_normal: Number of normal samples.
        n_anomaly: Number of anomalous samples.
        image_size: Image size (H, W).

    Returns:
        Tuple of (X_train, X_test, y_test).
    """
    print(f"Generating {n_normal} normal + {n_anomaly} anomaly samples...")

    # Normal images: simple patterns
    normal_images = []
    for _ in range(n_normal):
        img = np.random.rand(*image_size, 3) * 50  # Dark background
        # Add structured patterns
        img[::8, :, :] = 200  # Horizontal lines
        img[:, ::8, :] = 200  # Vertical lines
        normal_images.append(img.astype(np.uint8))

    # Test normal images
    test_normal = []
    for _ in range(n_anomaly):
        img = np.random.rand(*image_size, 3) * 50
        img[::8, :, :] = 200
        img[:, ::8, :] = 200
        test_normal.append(img.astype(np.uint8))

    # Anomalous images: broken patterns
    anomaly_images = []
    for _ in range(n_anomaly):
        img = np.random.rand(*image_size, 3) * 100  # Different intensity
        # Add random defects
        cx, cy = np.random.randint(50, 200, 2)
        cv2.circle(img, (cx, cy), 30, (255, 0, 0), -1)
        anomaly_images.append(img.astype(np.uint8))

    X_train = np.array(normal_images)
    X_test = np.array(test_normal + anomaly_images)
    y_test = np.array([0] * len(test_normal) + [1] * len(anomaly_images))

    return X_train, X_test, y_test


def demo_cutpaste():
    """Demonstrate CutPaste algorithm."""
    print("\n" + "=" * 70)
    print("CutPaste: Self-Supervised Learning (CVPR 2021)")
    print("=" * 70)

    # Generate data
    X_train, X_test, y_test = generate_sample_data(n_normal=50, n_anomaly=10)

    # Create CutPaste detector
    print("\nCreating CutPaste detector...")
    detector = create_model(
        "cutpaste",
        backbone="resnet18",
        augment_type="normal",  # "normal", "scar", or "3way"
        epochs=20,  # Fewer epochs for demo
        batch_size=8,
        learning_rate=0.03,
    )

    # Train
    print("Training CutPaste (this may take a few minutes)...")
    detector.fit(X_train)

    # Predict
    print("Predicting anomaly scores...")
    scores = detector.predict_proba(X_test)
    predictions = detector.predict(X_test)

    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score

    auc = roc_auc_score(y_test, scores)
    acc = accuracy_score(y_test, predictions)

    print(f"\nResults:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Normal samples detected: {sum(predictions[:10] == 0)}/10")
    print(f"  Anomaly samples detected: {sum(predictions[10:] == 1)}/10")

    return detector, scores, y_test


def demo_winclip():
    """Demonstrate WinCLIP algorithm."""
    print("\n" + "=" * 70)
    print("WinCLIP: Zero-Shot CLIP-based Detection (CVPR 2023)")
    print("=" * 70)

    try:
        import clip
    except ImportError:
        print("\nSkipping WinCLIP demo - CLIP not installed")
        print("Install with: pip install git+https://github.com/openai/CLIP.git")
        return None, None, None

    # Generate data
    X_train, X_test, y_test = generate_sample_data(n_normal=20, n_anomaly=10)

    # Create WinCLIP detector
    print("\nCreating WinCLIP detector...")
    detector = create_model(
        "winclip",
        clip_model="ViT-B/32",
        window_size=128,
        window_stride=64,
        k_shot=5,  # Use 5-shot learning
    )

    # Set class name for text prompts
    detector.set_class_name("grid pattern")

    # Train (few-shot)
    print("Training WinCLIP (few-shot learning)...")
    detector.fit(X_train)

    # Predict
    print("Predicting anomaly scores...")
    scores = detector.predict_proba(X_test)
    predictions = detector.predict(X_test)

    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score

    auc = roc_auc_score(y_test, scores)
    acc = accuracy_score(y_test, predictions)

    print(f"\nResults:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    # Demo anomaly localization
    print("\nGenerating anomaly maps for first 3 test images...")
    anomaly_maps = detector.predict_anomaly_map(X_test[:3])
    for i, map in enumerate(anomaly_maps):
        print(
            f"  Image {i}: Anomaly map shape {map.shape}, "
            f"max={map.max():.3f}, mean={map.mean():.3f}"
        )

    return detector, scores, y_test


def demo_differnet():
    """Demonstrate DifferNet algorithm."""
    print("\n" + "=" * 70)
    print("DifferNet: Learnable Difference Detection (WACV 2023)")
    print("=" * 70)

    # Generate data
    X_train, X_test, y_test = generate_sample_data(n_normal=50, n_anomaly=10)

    # Create DifferNet detector
    print("\nCreating DifferNet detector...")
    detector = create_model(
        "differnet",
        backbone="resnet18",  # Use smaller backbone for demo
        k_neighbors=3,
        feature_layer="layer3",
        train_difference=True,
        epochs=5,  # Fewer epochs for demo
        batch_size=8,
    )

    # Train
    print("Training DifferNet...")
    detector.fit(X_train)

    # Predict
    print("Predicting anomaly scores...")
    scores = detector.predict_proba(X_test)
    predictions = detector.predict(X_test)

    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score

    auc = roc_auc_score(y_test, scores)
    acc = accuracy_score(y_test, predictions)

    print(f"\nResults:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Memory bank size: {len(detector.memory_bank['layer3'])} features")

    return detector, scores, y_test


def compare_algorithms():
    """Compare all SOTA algorithms."""
    print("\n" + "=" * 70)
    print("Comparing SOTA Algorithms")
    print("=" * 70)

    # Generate shared test data
    X_train, X_test, y_test = generate_sample_data(n_normal=50, n_anomaly=10)

    results = {}

    # Test each algorithm
    algorithms = [
        ("CutPaste", demo_cutpaste),
        ("WinCLIP", demo_winclip),
        ("DifferNet", demo_differnet),
    ]

    for name, demo_func in algorithms:
        try:
            _, scores, _ = demo_func()
            if scores is not None:
                from sklearn.metrics import roc_auc_score

                auc = roc_auc_score(y_test, scores)
                results[name] = auc
        except Exception as e:
            print(f"\nError running {name}: {e}")
            results[name] = None

    # Print comparison
    print("\n" + "=" * 70)
    print("Algorithm Comparison (AUC-ROC)")
    print("=" * 70)
    for name, auc in results.items():
        if auc is not None:
            print(f"  {name:15s}: {auc:.4f}")
        else:
            print(f"  {name:15s}: N/A (error or skipped)")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("PyImgAno SOTA Algorithms Demo")
    print("=" * 70)
    print("\nThis script demonstrates the latest state-of-the-art algorithms:")
    print("  1. CutPaste (CVPR 2021) - Self-supervised learning")
    print("  2. WinCLIP (CVPR 2023) - Zero-shot CLIP-based detection")
    print("  3. DifferNet (WACV 2023) - Learnable difference detection")

    # Demo individual algorithms
    demo_cutpaste()
    demo_winclip()
    demo_differnet()

    # Compare algorithms
    print("\n\nRunning full comparison...")
    compare_algorithms()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
