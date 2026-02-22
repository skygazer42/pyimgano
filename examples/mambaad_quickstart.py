"""
MambaAD-style detector quickstart.

Prereqs:
  pip install -e ".[mamba]"   # or: pip install "pyimgano[mamba]"

Notes:
- The default embedder uses DINOv2 via torch.hub and may download weights on first run.
- For offline environments, pass a custom embedder via `embedder=...`.
"""

from __future__ import annotations

from pyimgano.models import create_model


def main() -> None:
    # Normal/reference images (paths). Replace with your own dataset.
    train_paths = [
        "normal_1.jpg",
        "normal_2.jpg",
    ]

    test_paths = [
        "test_1.jpg",
        "test_2.jpg",
    ]

    detector = create_model(
        "vision_mambaad",
        device="cuda",       # or "cpu"
        epochs=3,            # small demo value
        batch_size=4,
        lr=1e-3,
        image_size=518,      # DINOv2-friendly input size
        aggregation_method="topk_mean",
        aggregation_topk=0.01,
        contamination=0.1,
    )

    detector.fit(train_paths)
    scores = detector.decision_function(test_paths)
    labels = detector.predict(test_paths)

    print("scores:", scores)
    print("labels:", labels)

    # Pixel map (localization)
    heatmap = detector.get_anomaly_map(test_paths[0])
    print("anomaly_map:", heatmap.shape, heatmap.dtype)


if __name__ == "__main__":
    main()

