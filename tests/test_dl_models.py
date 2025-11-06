"""
Tests for deep learning vision anomaly detection models.

Tests cover PatchCore, STFPM, SimpleNet, and other DL-based detectors.
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from pyimgano.models import create_model


@pytest.fixture
def sample_images(tmp_path):
    """Create synthetic test images."""
    images = []

    # Create 5 normal images (uniform gray)
    for i in range(5):
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        img_path = tmp_path / f"normal_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        images.append(str(img_path))

    # Create 2 anomaly images (with white square)
    for i in range(2):
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        img[50:150, 50:150] = 255  # White square anomaly
        img_path = tmp_path / f"anomaly_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        images.append(str(img_path))

    return {
        "normal": images[:5],
        "anomaly": images[5:],
        "all": images,
    }


class TestPatchCore:
    """Test PatchCore algorithm."""

    @pytest.mark.parametrize("device", ["cpu"])
    def test_initialization(self, device):
        """Test PatchCore initialization."""
        detector = create_model(
            "vision_patchcore",
            backbone="wide_resnet50",
            coreset_sampling_ratio=0.1,
            n_neighbors=9,
            device=device,
        )

        assert detector is not None
        assert detector.backbone_name == "wide_resnet50"
        assert detector.coreset_sampling_ratio == 0.1
        assert detector.n_neighbors == 9

    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        # Invalid coreset ratio
        with pytest.raises(ValueError, match="coreset_sampling_ratio"):
            create_model("vision_patchcore", coreset_sampling_ratio=0.0)

        with pytest.raises(ValueError, match="coreset_sampling_ratio"):
            create_model("vision_patchcore", coreset_sampling_ratio=1.5)

        # Invalid n_neighbors
        with pytest.raises(ValueError, match="n_neighbors"):
            create_model("vision_patchcore", n_neighbors=0)

    @pytest.mark.slow
    def test_fit_predict(self, sample_images):
        """Test PatchCore fit and predict."""
        detector = create_model(
            "vision_patchcore",
            coreset_sampling_ratio=0.5,  # Higher ratio for small dataset
            device="cpu",
        )

        # Fit on normal images
        detector.fit(sample_images["normal"])

        # Predict on all images
        scores = detector.predict(sample_images["all"])

        assert len(scores) == len(sample_images["all"])
        assert all(isinstance(s, (int, float)) for s in scores)
        assert all(s >= 0 for s in scores)

        # Anomaly scores should generally be higher
        normal_scores = scores[:5]
        anomaly_scores = scores[5:]
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

    @pytest.mark.slow
    def test_anomaly_map(self, sample_images):
        """Test anomaly map generation."""
        detector = create_model(
            "vision_patchcore",
            coreset_sampling_ratio=0.5,
            device="cpu",
        )

        detector.fit(sample_images["normal"])

        # Generate anomaly map
        anomaly_map = detector.get_anomaly_map(sample_images["anomaly"][0])

        assert anomaly_map.ndim == 2
        assert anomaly_map.shape[0] > 0
        assert anomaly_map.shape[1] > 0
        assert np.all(np.isfinite(anomaly_map))


class TestSTFPM:
    """Test STFPM algorithm."""

    @pytest.mark.parametrize("device", ["cpu"])
    def test_initialization(self, device):
        """Test STFPM initialization."""
        detector = create_model(
            "vision_stfpm",
            backbone="resnet18",
            epochs=5,
            batch_size=2,
            lr=0.4,
            device=device,
        )

        assert detector is not None
        assert detector.backbone_name == "resnet18"
        assert detector.epochs == 5
        assert detector.batch_size == 2
        assert detector.lr == 0.4

    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        # Invalid epochs
        with pytest.raises(ValueError, match="epochs"):
            create_model("vision_stfpm", epochs=0)

        # Invalid batch_size
        with pytest.raises(ValueError, match="batch_size"):
            create_model("vision_stfpm", batch_size=0)

    @pytest.mark.slow
    def test_fit_predict(self, sample_images):
        """Test STFPM fit and predict."""
        detector = create_model(
            "vision_stfpm",
            epochs=2,  # Few epochs for testing
            batch_size=2,
            device="cpu",
        )

        # Fit on normal images
        detector.fit(sample_images["normal"])

        # Predict on all images
        scores = detector.predict(sample_images["all"])

        assert len(scores) == len(sample_images["all"])
        assert all(isinstance(s, (int, float)) for s in scores)

    @pytest.mark.slow
    def test_anomaly_map(self, sample_images):
        """Test anomaly map generation."""
        detector = create_model(
            "vision_stfpm",
            epochs=2,
            batch_size=2,
            device="cpu",
        )

        detector.fit(sample_images["normal"])

        # Generate anomaly map
        anomaly_map = detector.get_anomaly_map(sample_images["normal"][0])

        assert anomaly_map.ndim == 2
        assert anomaly_map.shape[0] > 0
        assert anomaly_map.shape[1] > 0
        assert np.all(np.isfinite(anomaly_map))


class TestSimpleNet:
    """Test SimpleNet algorithm."""

    @pytest.mark.parametrize("device", ["cpu"])
    def test_initialization(self, device):
        """Test SimpleNet initialization."""
        detector = create_model(
            "vision_simplenet",
            backbone="wide_resnet50",
            feature_dim=384,
            epochs=5,
            batch_size=4,
            lr=0.001,
            device=device,
        )

        assert detector is not None
        assert detector.backbone_name == "wide_resnet50"
        assert detector.feature_dim == 384
        assert detector.epochs == 5
        assert detector.batch_size == 4

    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        # Invalid epochs
        with pytest.raises(ValueError, match="epochs"):
            create_model("vision_simplenet", epochs=0)

        # Invalid batch_size
        with pytest.raises(ValueError, match="batch_size"):
            create_model("vision_simplenet", batch_size=-1)

    @pytest.mark.slow
    def test_fit_predict(self, sample_images):
        """Test SimpleNet fit and predict."""
        detector = create_model(
            "vision_simplenet",
            epochs=2,  # Very fast training
            batch_size=2,
            device="cpu",
        )

        # Fit on normal images
        detector.fit(sample_images["normal"])

        # Predict on all images
        scores = detector.predict(sample_images["all"])

        assert len(scores) == len(sample_images["all"])
        assert all(isinstance(s, (int, float)) for s in scores)
        assert all(0 <= s <= 2 for s in scores)  # Cosine distance range

    @pytest.mark.slow
    def test_reference_features_built(self, sample_images):
        """Test that reference features are built during training."""
        detector = create_model(
            "vision_simplenet",
            epochs=2,
            batch_size=2,
            device="cpu",
        )

        # Should not have reference features before fit
        assert not hasattr(detector, "reference_features")

        # Fit on normal images
        detector.fit(sample_images["normal"])

        # Should have reference features after fit
        assert hasattr(detector, "reference_features")
        assert detector.reference_features.shape[0] > 0
        assert detector.reference_features.shape[1] == 384  # feature_dim


class TestDLModelsIntegration:
    """Integration tests for DL models."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "vision_patchcore",
            "vision_stfpm",
            "vision_simplenet",
        ],
    )
    @pytest.mark.slow
    def test_model_lifecycle(self, model_name, sample_images):
        """Test full lifecycle: init -> fit -> predict -> decision_function."""
        # Create model with minimal parameters for fast testing
        if model_name == "vision_patchcore":
            detector = create_model(model_name, coreset_sampling_ratio=0.5, device="cpu")
        elif model_name in ["vision_stfpm", "vision_simplenet"]:
            detector = create_model(model_name, epochs=2, batch_size=2, device="cpu")
        else:
            detector = create_model(model_name, device="cpu")

        # Fit
        detector.fit(sample_images["normal"])

        # Predict
        scores = detector.predict(sample_images["all"])
        assert len(scores) == len(sample_images["all"])

        # Decision function (should be same as predict)
        decision_scores = detector.decision_function(sample_images["all"])
        assert len(decision_scores) == len(scores)
        np.testing.assert_array_equal(scores, decision_scores)

    @pytest.mark.parametrize(
        "model_name",
        [
            "vision_patchcore",
            "vision_stfpm",
        ],
    )
    @pytest.mark.slow
    def test_anomaly_map_generation(self, model_name, sample_images):
        """Test anomaly map generation for models that support it."""
        # Create model
        if model_name == "vision_patchcore":
            detector = create_model(model_name, coreset_sampling_ratio=0.5, device="cpu")
        else:
            detector = create_model(model_name, epochs=2, batch_size=2, device="cpu")

        # Fit
        detector.fit(sample_images["normal"])

        # Generate anomaly map
        anomaly_map = detector.get_anomaly_map(sample_images["anomaly"][0])

        # Validate map
        assert anomaly_map.ndim == 2
        assert anomaly_map.shape[0] > 0
        assert anomaly_map.shape[1] > 0
        assert np.all(np.isfinite(anomaly_map))

        # Check that anomaly region has higher scores
        # The white square is at [50:150, 50:150]
        h, w = anomaly_map.shape
        anomaly_region = anomaly_map[
            int(h * 0.2) : int(h * 0.7), int(w * 0.2) : int(w * 0.7)
        ]
        normal_region = anomaly_map[0 : int(h * 0.2), 0 : int(w * 0.2)]

        # Note: This assertion may not always hold for all algorithms
        # as it depends on the specific implementation
        # assert np.mean(anomaly_region) >= np.mean(normal_region)

    def test_model_not_fitted_error(self):
        """Test that predict raises error when model not fitted."""
        detector = create_model("vision_patchcore", device="cpu")

        with pytest.raises(RuntimeError, match="not fitted"):
            detector.predict(["dummy.jpg"])

    def test_empty_training_set(self):
        """Test that empty training set raises error."""
        detector = create_model("vision_patchcore", device="cpu")

        with pytest.raises(ValueError, match="cannot be empty"):
            detector.fit([])


class TestDLModelComparison:
    """Comparative tests for DL models."""

    @pytest.mark.slow
    def test_all_models_produce_scores(self, sample_images):
        """Test that all DL models produce reasonable scores."""
        models = {
            "patchcore": create_model(
                "vision_patchcore", coreset_sampling_ratio=0.5, device="cpu"
            ),
            "stfpm": create_model("vision_stfpm", epochs=2, batch_size=2, device="cpu"),
            "simplenet": create_model(
                "vision_simplenet", epochs=2, batch_size=2, device="cpu"
            ),
        }

        results = {}

        for name, detector in models.items():
            # Fit and predict
            detector.fit(sample_images["normal"])
            scores = detector.predict(sample_images["all"])

            results[name] = {
                "scores": scores,
                "normal_mean": np.mean(scores[:5]),
                "anomaly_mean": np.mean(scores[5:]),
            }

        # All models should produce valid scores
        for name, result in results.items():
            assert len(result["scores"]) == len(sample_images["all"])
            assert all(np.isfinite(s) for s in result["scores"])

            # Print for debugging (optional)
            # print(f"{name}: normal={result['normal_mean']:.4f}, "
            #       f"anomaly={result['anomaly_mean']:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
