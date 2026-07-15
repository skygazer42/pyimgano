import inspect

import numpy as np
import pytest

from pyimgano.models import create_model


class _FakePatchEmbedder:
    def embed(self, image_path: str):
        grid_h, grid_w = 2, 2
        original_h, original_w = 8, 8

        if "anomaly" in image_path:
            patch_embeddings = np.array(
                [
                    [10.0, 10.0],
                    [0.0, 0.0],
                    [10.0, 10.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            )
        else:
            patch_embeddings = np.zeros((grid_h * grid_w, 2), dtype=np.float32)

        return patch_embeddings, (grid_h, grid_w), (original_h, original_w)


def test_softpatch_registry_and_basic_api():
    det = create_model(
        "vision_softpatch",
        embedder=_FakePatchEmbedder(),
        contamination=0.1,
        knn_backend="sklearn",
        n_neighbors=1,
        aggregation_method="topk_mean",
        aggregation_topk=0.25,
    )

    det.fit(["train_1.png", "train_2.png"])

    scores = det.decision_function(["normal.png", "anomaly.png"])
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])

    anomaly_map = det.get_anomaly_map("anomaly.png")
    assert anomaly_map.shape == (8, 8)
    assert np.isfinite(anomaly_map).all()


def test_softpatch_paper_defaults():
    from pyimgano.models.softpatch import VisionSoftPatch

    parameters = inspect.signature(VisionSoftPatch).parameters
    assert parameters["backbone"].default == "wide_resnet50_2"
    assert parameters["layers"].default == ("layer2", "layer3")
    assert parameters["coreset_sampling_ratio"].default == pytest.approx(0.1)
    assert parameters["coreset_projection_dim"].default == 128
    assert parameters["coreset_starting_points"].default == 10
    assert parameters["n_neighbors"].default == 1
    assert parameters["weight_method"].default == "lof"
    assert parameters["lof_k"].default == 6
    assert parameters["train_patch_outlier_quantile"].default == pytest.approx(0.15)
    assert parameters["soft_weight"].default is True
    assert parameters["noise_projection_dim"].default == 128
    assert parameters["gaussian_regularization"].default == pytest.approx(0.01)
    assert parameters["pretrain_embed_dimension"].default == 1024
    assert parameters["target_embed_dimension"].default == 1024
    assert parameters["patch_size"].default == 3
    assert parameters["patch_stride"].default == 1
    assert parameters["resize_size"].default == 256
    assert parameters["image_size"].default == 224
    assert parameters["aggregation_method"].default == "max"
    assert parameters["gaussian_sigma"].default == pytest.approx(4.0)


def test_softpatch_robust_memory_filtering_reduces_bank():
    class _OutlierEmbedder:
        def embed(self, image_path: str):
            grid_h, grid_w = 2, 2
            original_h, original_w = 8, 8
            value = (
                0.07
                if "noisy" in image_path
                else 0.01 * int(image_path.split("_")[-1].split(".")[0])
            )
            patch_embeddings = np.full((grid_h * grid_w, 2), value, dtype=np.float32)
            patch_embeddings[1] = np.array([50.0 + value, 50.0 + value], dtype=np.float32)
            if "noisy" in image_path:
                # Inject two extreme outlier patches.
                patch_embeddings[0] = np.array([100.0, 100.0], dtype=np.float32)
                patch_embeddings[3] = np.array([100.0, 100.0], dtype=np.float32)
            return patch_embeddings, (grid_h, grid_w), (original_h, original_w)

    det_plain = create_model(
        "vision_softpatch",
        embedder=_OutlierEmbedder(),
        train_patch_outlier_quantile=0.0,
        coreset_sampling_ratio=1.0,
    )
    training = [f"clean_{index}.png" for index in range(7)] + ["noisy_0.png"]
    det_plain.fit(training)

    det_filtered = create_model(
        "vision_softpatch",
        embedder=_OutlierEmbedder(),
        train_patch_outlier_quantile=0.25,
        coreset_sampling_ratio=0.25,
    )
    det_filtered.fit(training)

    assert det_filtered.memory_bank_size_ == 8
    assert det_filtered.filtered_patches_ == 8
    assert float(np.max(det_filtered._memory_bank)) < 100.0
    assert float(np.max(det_filtered._memory_bank)) > 49.0


def test_softpatch_multiplies_nearest_distance_by_memory_outlier_weight():
    det = create_model(
        "vision_softpatch",
        embedder=_FakePatchEmbedder(),
        train_patch_outlier_quantile=0.0,
        coreset_sampling_ratio=1.0,
        aggregation_method="max",
        gaussian_sigma=0.0,
    )
    det.fit(["train_1.png", "train_2.png"])
    det._memory_bank_weights[:] = 3.0

    score = float(det.decision_function(["anomaly.png"])[0])
    assert score == pytest.approx(3.0 * np.sqrt(200.0), rel=1e-6)


@pytest.mark.parametrize("weight_method", ["nearest", "gaussian"])
def test_softpatch_paper_discriminator_scores(weight_method):
    det = create_model(
        "vision_softpatch",
        embedder=_FakePatchEmbedder(),
        weight_method=weight_method,
        noise_projection_dim=1,
        gaussian_regularization=0.01,
    )
    features = np.asarray([[[0.0]], [[1.0]], [[4.0]]], dtype=np.float32)

    weights = det._compute_patch_outlier_weights(features).reshape(-1)
    if weight_method == "nearest":
        np.testing.assert_allclose(weights, [2.0, 2.0, 4.0], rtol=1e-6, atol=1e-6)
    else:
        centered = np.asarray([-5.0 / 3.0, -2.0 / 3.0, 7.0 / 3.0])
        expected = 1.0 + np.abs(centered) / np.sqrt(np.sum(centered**2) / 2.0 + 0.01)
        np.testing.assert_allclose(weights, expected, rtol=1e-5, atol=1e-5)
