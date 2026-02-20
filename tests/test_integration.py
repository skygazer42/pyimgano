"""
End-to-end integration tests for PyImgAno.

These tests verify that the complete workflow works correctly.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from pyimgano import (
    AlgorithmBenchmark,
    compute_auroc,
    evaluate_detector,
    models,
    quick_benchmark,
)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create synthetic image dataset for testing."""
    # Create directory structure
    train_dir = tmp_path / "train"
    test_normal_dir = tmp_path / "test" / "normal"
    test_anomaly_dir = tmp_path / "test" / "anomaly"

    train_dir.mkdir(parents=True)
    test_normal_dir.mkdir(parents=True)
    test_anomaly_dir.mkdir(parents=True)

    def create_image(path, is_anomaly=False):
        """Create synthetic image."""
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128

        if is_anomaly:
            # Add white square as anomaly
            img[80:140, 80:140] = 255

        cv2.imwrite(str(path), img)

    # Create training images (normal only)
    train_images = []
    for i in range(10):
        img_path = train_dir / f"train_{i}.jpg"
        create_image(img_path, is_anomaly=False)
        train_images.append(str(img_path))

    # Create test images
    test_normal = []
    for i in range(5):
        img_path = test_normal_dir / f"normal_{i}.jpg"
        create_image(img_path, is_anomaly=False)
        test_normal.append(str(img_path))

    test_anomaly = []
    for i in range(5):
        img_path = test_anomaly_dir / f"anomaly_{i}.jpg"
        create_image(img_path, is_anomaly=True)
        test_anomaly.append(str(img_path))

    return {
        'train': train_images,
        'test_normal': test_normal,
        'test_anomaly': test_anomaly,
        'test_all': test_normal + test_anomaly,
        'test_labels': np.array([0] * len(test_normal) + [1] * len(test_anomaly)),
    }


class TestBasicWorkflow:
    """Test basic anomaly detection workflow."""

    @pytest.mark.parametrize(
        "model_name",
        ["vision_ecod", "vision_copod", "vision_knn"]
    )
    def test_fit_predict_workflow(self, synthetic_dataset, model_name):
        """Test basic fit-predict workflow."""
        # Create detector
        detector = models.create_model(model_name, contamination=0.1)

        # Fit
        detector.fit(synthetic_dataset['train'])

        # Get continuous anomaly scores
        scores = detector.decision_function(synthetic_dataset['test_all'])

        # Verify
        assert len(scores) == len(synthetic_dataset['test_all'])
        assert all(isinstance(s, (int, float)) for s in scores)
        assert all(np.isfinite(s) for s in scores)

    def test_evaluation_workflow(self, synthetic_dataset):
        """Test complete evaluation workflow."""
        # Train detector
        detector = models.create_model('vision_ecod', contamination=0.1)
        detector.fit(synthetic_dataset['train'])

        # Get scores
        scores = detector.decision_function(synthetic_dataset['test_all'])

        # Evaluate
        results = evaluate_detector(
            synthetic_dataset['test_labels'],
            scores
        )

        # Verify results structure
        assert 'auroc' in results
        assert 'average_precision' in results
        assert 'threshold' in results
        assert 'metrics' in results

        # Verify metrics are valid
        assert 0 <= results['auroc'] <= 1
        assert 0 <= results['average_precision'] <= 1
        assert 0 <= results['metrics']['f1'] <= 1


class TestBenchmarkWorkflow:
    """Test benchmarking functionality."""

    def test_algorithm_benchmark(self, synthetic_dataset):
        """Test AlgorithmBenchmark class."""
        algorithms = {
            'ECOD': {'model_name': 'vision_ecod', 'contamination': 0.1},
            'COPOD': {'model_name': 'vision_copod', 'contamination': 0.1},
        }

        benchmark = AlgorithmBenchmark(algorithms)
        results = benchmark.run(
            train_images=synthetic_dataset['train'],
            test_images=synthetic_dataset['test_all'],
            test_labels=synthetic_dataset['test_labels'],
            verbose=False
        )

        # Verify results
        assert len(results) == 2
        assert 'ECOD' in results
        assert 'COPOD' in results

        for algo_name, result in results.items():
            assert result['success']
            assert 'auroc' in result
            assert 'train_time' in result
            assert 'inference_time' in result

    def test_quick_benchmark(self, synthetic_dataset):
        """Test quick_benchmark helper function."""
        results = quick_benchmark(
            train_images=synthetic_dataset['train'],
            test_images=synthetic_dataset['test_all'],
            test_labels=synthetic_dataset['test_labels'],
            algorithms=['ECOD', 'COPOD']
        )

        assert len(results) >= 2
        for result in results.values():
            if result.get('success'):
                assert 'auroc' in result


class TestDeepLearningWorkflow:
    """Test deep learning models (marked as slow)."""

    @pytest.mark.slow
    def test_simplenet_workflow(self, synthetic_dataset):
        """Test SimpleNet end-to-end."""
        detector = models.create_model(
            'vision_simplenet',
            epochs=2,  # Minimal for testing
            batch_size=2,
            device='cpu'
        )

        # Train
        detector.fit(synthetic_dataset['train'])

        # Predict
        scores = detector.decision_function(synthetic_dataset['test_all'])

        # Evaluate
        auroc = compute_auroc(synthetic_dataset['test_labels'], scores)

        assert 0 <= auroc <= 1

    @pytest.mark.slow
    def test_patchcore_workflow(self, synthetic_dataset):
        """Test PatchCore end-to-end."""
        detector = models.create_model(
            'vision_patchcore',
            coreset_sampling_ratio=0.5,
            device='cpu'
        )

        # Train (feature extraction only)
        detector.fit(synthetic_dataset['train'])

        # Predict
        scores = detector.decision_function(synthetic_dataset['test_all'])

        # Evaluate
        results = evaluate_detector(
            synthetic_dataset['test_labels'],
            scores
        )

        assert 0 <= results['auroc'] <= 1

        # Test anomaly map generation
        anomaly_map = detector.get_anomaly_map(
            synthetic_dataset['test_anomaly'][0]
        )

        assert anomaly_map.ndim == 2
        assert anomaly_map.shape[0] > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_training_set(self):
        """Test that empty training set raises error."""
        detector = models.create_model('vision_ecod')

        with pytest.raises(ValueError, match="cannot be empty"):
            detector.fit([])

    def test_invalid_image_path(self, synthetic_dataset):
        """Test handling of invalid image paths."""
        detector = models.create_model('vision_ecod')
        detector.fit(synthetic_dataset['train'])

        # This should handle the error gracefully
        scores = detector.decision_function(['nonexistent.jpg'])
        # Typically returns 0.0 for failed images
        assert len(scores) == 1

    def test_model_not_fitted(self):
        """Test that predict before fit raises error."""
        detector = models.create_model('vision_patchcore')

        with pytest.raises(RuntimeError, match="not fitted"):
            detector.predict(['test.jpg'])


class TestSaveLoad:
    """Test model persistence (if supported)."""

    def test_pickle_detector(self, synthetic_dataset, tmp_path):
        """Test saving and loading detector with pickle."""
        import pickle

        # Train detector
        detector = models.create_model('vision_ecod', contamination=0.1)
        detector.fit(synthetic_dataset['train'])

        # Get original predictions
        original_scores = detector.decision_function(synthetic_dataset['test_all'])

        # Save
        model_path = tmp_path / 'detector.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(detector, f)

        # Load
        with open(model_path, 'rb') as f:
            loaded_detector = pickle.load(f)

        # Get loaded predictions
        loaded_scores = loaded_detector.decision_function(synthetic_dataset['test_all'])

        # Compare
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
