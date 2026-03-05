"""
Benchmark Example for PyImgAno.

This script demonstrates how to benchmark multiple algorithms.
"""

from pathlib import Path

import numpy as np

from pyimgano import AlgorithmBenchmark


def main():
    """Run benchmark example."""
    print("=" * 60)
    print("PyImgAno Benchmark Example")
    print("=" * 60 + "\n")

    # Setup paths (adjust to your dataset)
    train_dir = Path("data/train/normal")
    test_normal_dir = Path("data/test/normal")
    test_anomaly_dir = Path("data/test/anomaly")

    if not train_dir.exists():
        print("⚠️  Data directory not found. Please set up your dataset first.")
        print("See examples/quick_start.py for instructions.")
        return

    # Collect images
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    test_normal = list(test_normal_dir.glob("*.jpg")) + list(test_normal_dir.glob("*.png"))
    test_anomaly = list(test_anomaly_dir.glob("*.jpg")) + list(test_anomaly_dir.glob("*.png"))

    train_images = [str(p) for p in train_images]
    test_images = [str(p) for p in test_normal + test_anomaly]
    test_labels = np.array([0] * len(test_normal) + [1] * len(test_anomaly))

    print(f"Dataset: {len(train_images)} train, {len(test_images)} test images\n")

    # Define algorithms to benchmark
    algorithms = {
        "ECOD": {
            "model_name": "vision_ecod",
            "contamination": 0.1,
        },
        "COPOD": {
            "model_name": "vision_copod",
            "contamination": 0.1,
        },
        "KNN": {
            "model_name": "vision_knn",
            "n_neighbors": 5,
            "contamination": 0.1,
        },
        "PCA": {
            "model_name": "vision_pca",
            "contamination": 0.1,
        },
    }

    # Create and run benchmark
    benchmark = AlgorithmBenchmark(algorithms)
    results = benchmark.run(
        train_images=train_images, test_images=test_images, test_labels=test_labels, verbose=True
    )

    # Print summary
    benchmark.print_summary()

    # Get rankings
    print("\nRankings by AUROC:")
    for rank, (algo, auroc) in enumerate(benchmark.get_rankings("auroc"), 1):
        print(f"  {rank}. {algo}: {auroc:.4f}")

    print("\nRankings by Speed (inference per image):")
    for rank, (algo, time_ms) in enumerate(benchmark.get_rankings("inference_per_image"), 1):
        print(f"  {rank}. {algo}: {time_ms * 1000:.1f}ms")

    # Save results
    benchmark.save_results("benchmark_results.json")
    print("\n✓ Results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
