"""
Tests for evaluation metrics module.
"""

import numpy as np
import pytest

from pyimgano.evaluation import (
    compute_auroc,
    compute_average_precision,
    compute_classification_metrics,
    compute_pro_score,
    evaluate_detector,
    find_optimal_threshold,
)


class TestAUROC:
    """Test AUROC computation."""

    def test_perfect_separation(self):
        """Test AUROC with perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auroc = compute_auroc(y_true, y_scores)
        assert auroc == 1.0

    def test_random_classifier(self):
        """Test AUROC with random scores (around 0.5)."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 100)
        y_scores = np.random.rand(200)

        auroc = compute_auroc(y_true, y_scores)
        assert 0.4 <= auroc <= 0.6  # Should be around 0.5

    def test_single_class(self):
        """Test AUROC with only one class (undefined)."""
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])

        auroc = compute_auroc(y_true, y_scores)
        assert np.isnan(auroc)

    def test_worst_case(self):
        """Test AUROC with worst possible predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        auroc = compute_auroc(y_true, y_scores)
        assert auroc == 0.0


class TestAveragePrecision:
    """Test Average Precision computation."""

    def test_perfect_precision(self):
        """Test AP with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        ap = compute_average_precision(y_true, y_scores)
        assert ap == 1.0

    def test_single_class(self):
        """Test AP with only one class."""
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])

        ap = compute_average_precision(y_true, y_scores)
        assert np.isnan(ap)


class TestFindOptimalThreshold:
    """Test optimal threshold finding."""

    def test_f1_optimization(self):
        """Test finding optimal threshold for F1 score."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9])

        threshold, f1 = find_optimal_threshold(y_true, y_scores, metric='f1')

        assert 0 <= threshold <= 1
        assert 0 <= f1 <= 1

    def test_youden_optimization(self):
        """Test finding optimal threshold using Youden's J statistic."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9])

        threshold, j = find_optimal_threshold(y_true, y_scores, metric='youden')

        assert 0 <= threshold <= 1
        assert -1 <= j <= 1

    def test_precision_optimization(self):
        """Test finding optimal threshold for precision."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        threshold, precision = find_optimal_threshold(y_true, y_scores, metric='precision')

        assert 0 <= threshold <= 1
        assert 0 <= precision <= 1

    def test_recall_optimization(self):
        """Test finding optimal threshold for recall."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        threshold, recall = find_optimal_threshold(y_true, y_scores, metric='recall')

        assert 0 <= threshold <= 1
        assert 0 <= recall <= 1

    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        with pytest.raises(ValueError, match="Unsupported metric"):
            find_optimal_threshold(y_true, y_scores, metric='invalid')


class TestClassificationMetrics:
    """Test classification metrics computation."""

    def test_perfect_classification(self):
        """Test metrics with perfect classification."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['specificity'] == 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['tp'] == 2
        assert metrics['tn'] == 2
        assert metrics['fp'] == 0
        assert metrics['fn'] == 0

    def test_all_negative_predictions(self):
        """Test metrics when all predictions are negative."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['specificity'] == 1.0
        assert metrics['accuracy'] == 0.5
        assert metrics['tp'] == 0
        assert metrics['fn'] == 2

    def test_all_positive_predictions(self):
        """Test metrics when all predictions are positive."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics['recall'] == 1.0
        assert metrics['specificity'] == 0.0
        assert metrics['accuracy'] == 0.5


class TestEvaluateDetector:
    """Test comprehensive detector evaluation."""

    def test_with_explicit_threshold(self):
        """Test evaluation with explicit threshold."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        results = evaluate_detector(y_true, y_scores, threshold=0.5)

        assert 'auroc' in results
        assert 'average_precision' in results
        assert 'threshold' in results
        assert 'metrics' in results

        assert results['threshold'] == 0.5
        assert 0 <= results['auroc'] <= 1
        assert 0 <= results['average_precision'] <= 1

    def test_with_optimal_threshold(self):
        """Test evaluation with automatic threshold finding."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        results = evaluate_detector(y_true, y_scores, find_best_threshold=True)

        assert results['threshold'] is not None
        assert 0 <= results['threshold'] <= 1

    def test_classification_metrics_included(self):
        """Test that classification metrics are included."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        results = evaluate_detector(y_true, y_scores)

        metrics = results['metrics']
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'specificity' in metrics
        assert 'accuracy' in metrics
        assert 'tp' in metrics
        assert 'tn' in metrics
        assert 'fp' in metrics
        assert 'fn' in metrics

    def test_pixel_metrics_optional(self):
        """Test that pixel-level metrics are computed when provided."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        pixel_labels = np.zeros((1, 10, 10), dtype=np.uint8)
        pixel_labels[:, 2:5, 2:5] = 1
        pixel_scores = pixel_labels.astype(np.float32)

        results = evaluate_detector(
            y_true,
            y_scores,
            pixel_labels=pixel_labels,
            pixel_scores=pixel_scores,
        )

        assert 'pixel_metrics' in results
        pixel_metrics = results['pixel_metrics']
        assert 'pixel_auroc' in pixel_metrics
        assert 'pixel_average_precision' in pixel_metrics
        assert 'aupro' in pixel_metrics

    def test_pixel_segf1_optional_with_fixed_threshold(self):
        """Test SegF1/bg-FPR metrics under a single global pixel threshold."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        pixel_labels = np.zeros((1, 10, 10), dtype=np.uint8)
        pixel_labels[:, 2:5, 2:5] = 1
        pixel_scores = pixel_labels.astype(np.float32)

        results = evaluate_detector(
            y_true,
            y_scores,
            pixel_labels=pixel_labels,
            pixel_scores=pixel_scores,
            pixel_threshold=0.5,
        )

        pixel_metrics = results["pixel_metrics"]
        assert pixel_metrics["pixel_segf1"] == 1.0
        assert pixel_metrics["bg_fpr"] == 0.0


class TestPROScore:
    """Test PRO score computation."""

    def test_perfect_localization(self):
        """Test PRO score with perfect localization."""
        # Create synthetic pixel-level data
        pixel_labels = np.zeros((2, 100, 100))
        pixel_labels[:, 25:75, 25:75] = 1  # Anomalous region

        pixel_scores = pixel_labels.copy()  # Perfect prediction

        pro_score = compute_pro_score(pixel_labels, pixel_scores)

        assert 0 <= pro_score <= 1
        assert pro_score > 0.9  # Should be very high

    def test_random_localization(self):
        """Test PRO score with random predictions."""
        np.random.seed(42)
        pixel_labels = np.zeros((2, 50, 50))
        pixel_labels[:, 10:40, 10:40] = 1

        pixel_scores = np.random.rand(2, 50, 50)

        pro_score = compute_pro_score(pixel_labels, pixel_scores)

        assert 0 <= pro_score <= 1
        assert 0.2 <= pro_score <= 0.8  # Should be around 0.5

    def test_custom_integration_limit(self):
        """Test PRO score with custom integration limit."""
        pixel_labels = np.zeros((1, 50, 50))
        pixel_labels[:, 10:40, 10:40] = 1
        pixel_scores = pixel_labels.copy()

        pro_score = compute_pro_score(
            pixel_labels,
            pixel_scores,
            integration_limit=0.5
        )

        assert 0 <= pro_score <= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        y_true = np.array([])
        y_scores = np.array([])

        with pytest.raises(ValueError):
            compute_auroc(y_true, y_scores)

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        y_true = np.array([0, 1, 0])
        y_scores = np.array([0.1, 0.9])

        with pytest.raises(ValueError):
            compute_auroc(y_true, y_scores)

    def test_nan_in_scores(self):
        """Test handling of NaN values in scores."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, np.nan, 0.8, 0.9])

        # Should handle gracefully or raise appropriate error
        try:
            auroc = compute_auroc(y_true, y_scores)
            # If it returns a value, it should be NaN
            assert np.isnan(auroc) or 0 <= auroc <= 1
        except (ValueError, RuntimeError):
            # Also acceptable to raise an error
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
