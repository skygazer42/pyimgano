"""
Evaluation metrics for anomaly detection.

This module provides comprehensive evaluation metrics including:
- ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- Precision, Recall, F1-Score
- Average Precision (AP)
- Optimal threshold detection
- Confusion matrix utilities
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_auroc(
    y_true: NDArray,
    y_scores: NDArray,
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels (0 = normal, 1 = anomaly)
    y_scores : ndarray of shape (n_samples,)
        Anomaly scores (higher = more anomalous)

    Returns
    -------
    auroc : float
        ROC-AUC score in range [0, 1]

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    >>> auroc = compute_auroc(y_true, y_scores)
    >>> print(f"AUROC: {auroc:.3f}")
    AUROC: 1.000
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    if y_true.size == 0 or y_scores.size == 0:
        raise ValueError("y_true and y_scores must be non-empty.")
    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError(
            "y_true and y_scores must have the same length. "
            f"Got {y_true.shape[0]} != {y_scores.shape[0]}."
        )

    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true. AUROC is not defined.")
        return float('nan')

    try:
        auroc = roc_auc_score(y_true, y_scores)
        return float(auroc)
    except Exception as e:
        logger.error("Failed to compute AUROC: %s", e)
        return float('nan')


def compute_average_precision(
    y_true: NDArray,
    y_scores: NDArray,
) -> float:
    """
    Compute Average Precision (AP).

    AP summarizes the precision-recall curve as the weighted mean of
    precisions achieved at each threshold.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels (0 = normal, 1 = anomaly)
    y_scores : ndarray of shape (n_samples,)
        Anomaly scores (higher = more anomalous)

    Returns
    -------
    ap : float
        Average precision score

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    >>> ap = compute_average_precision(y_true, y_scores)
    >>> print(f"AP: {ap:.3f}")
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    if y_true.size == 0 or y_scores.size == 0:
        raise ValueError("y_true and y_scores must be non-empty.")
    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError(
            "y_true and y_scores must have the same length. "
            f"Got {y_true.shape[0]} != {y_scores.shape[0]}."
        )

    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true. AP is not defined.")
        return float('nan')

    try:
        ap = average_precision_score(y_true, y_scores)
        return float(ap)
    except Exception as e:
        logger.error("Failed to compute AP: %s", e)
        return float('nan')


def find_optimal_threshold(
    y_true: NDArray,
    y_scores: NDArray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find optimal threshold for classification.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels (0 = normal, 1 = anomaly)
    y_scores : ndarray of shape (n_samples,)
        Anomaly scores (higher = more anomalous)
    metric : str, default='f1'
        Metric to optimize ('f1', 'precision', 'recall', 'youden')

    Returns
    -------
    threshold : float
        Optimal threshold value
    metric_value : float
        Value of the metric at optimal threshold

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    >>> threshold, f1 = find_optimal_threshold(y_true, y_scores, metric='f1')
    >>> print(f"Optimal threshold: {threshold:.3f}, F1: {f1:.3f}")
    """
    if metric == "f1":
        # Optimize F1 score
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        # Compute F1 scores (avoid division by zero)
        f1_scores = np.zeros_like(precisions)
        mask = (precisions + recalls) > 0
        f1_scores[mask] = 2 * (precisions[mask] * recalls[mask]) / (
            precisions[mask] + recalls[mask]
        )

        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        optimal_threshold = float(thresholds[optimal_idx])
        optimal_f1 = float(f1_scores[optimal_idx])

        return optimal_threshold, optimal_f1

    elif metric == "youden":
        # Youden's J statistic = Sensitivity + Specificity - 1
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = float(thresholds[optimal_idx])
        optimal_j = float(youden_j[optimal_idx])

        return optimal_threshold, optimal_j

    elif metric in ["precision", "recall"]:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        if metric == "precision":
            optimal_idx = np.argmax(precisions[:-1])
            optimal_threshold = float(thresholds[optimal_idx])
            return optimal_threshold, float(precisions[optimal_idx])
        else:  # recall
            optimal_idx = np.argmax(recalls[:-1])
            optimal_threshold = float(thresholds[optimal_idx])
            return optimal_threshold, float(recalls[optimal_idx])

    else:
        raise ValueError(
            f"Unsupported metric: {metric}. "
            "Choose from 'f1', 'precision', 'recall', 'youden'"
        )


def compute_classification_metrics(
    y_true: NDArray,
    y_pred: NDArray,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels (0 = normal, 1 = anomaly)
    y_pred : ndarray of shape (n_samples,)
        Predicted binary labels

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'precision': Precision score
        - 'recall': Recall score (sensitivity)
        - 'f1': F1 score
        - 'specificity': Specificity (true negative rate)
        - 'accuracy': Accuracy score

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 1, 1])
    >>> metrics = compute_classification_metrics(y_true, y_pred)
    >>> print(f"F1: {metrics['f1']:.3f}")
    """
    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Specificity and accuracy
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def evaluate_detector(
    y_true: NDArray,
    y_scores: NDArray,
    threshold: Optional[float] = None,
    find_best_threshold: bool = True,
    pixel_labels: Optional[NDArray] = None,
    pixel_scores: Optional[NDArray] = None,
    pro_integration_limit: float = 0.3,
) -> Dict[str, Union[float, Dict]]:
    """
    Comprehensive evaluation of anomaly detector.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels (0 = normal, 1 = anomaly)
    y_scores : ndarray of shape (n_samples,)
        Anomaly scores (higher = more anomalous)
    threshold : float, optional
        Classification threshold. If None, will find optimal threshold.
    find_best_threshold : bool, default=True
        Whether to find optimal threshold when threshold is None

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'auroc': ROC-AUC score
        - 'average_precision': Average precision score
        - 'threshold': Classification threshold used
        - 'metrics': Classification metrics (precision, recall, f1, etc.)

    Examples
    --------
    >>> from pyimgano.models import create_model
    >>> detector = create_model('vision_ecod')
    >>> detector.fit(train_images)
    >>> scores = detector.decision_function(test_images)
    >>> results = evaluate_detector(test_labels, scores)
    >>> print(f"AUROC: {results['auroc']:.3f}")
    >>> print(f"F1: {results['metrics']['f1']:.3f}")
    """
    logger.info("Evaluating anomaly detector on %d samples", len(y_true))

    # Compute score-based metrics
    auroc = compute_auroc(y_true, y_scores)
    ap = compute_average_precision(y_true, y_scores)

    logger.info("AUROC: %.4f, AP: %.4f", auroc, ap)

    # Determine threshold
    if threshold is None and find_best_threshold:
        threshold, f1_at_threshold = find_optimal_threshold(
            y_true, y_scores, metric='f1'
        )
        logger.info(
            "Optimal threshold: %.4f (F1=%.4f)",
            threshold, f1_at_threshold
        )
    elif threshold is None:
        # Use median as default threshold
        threshold = float(np.median(y_scores))
        logger.info("Using median threshold: %.4f", threshold)

    # Convert scores to predictions
    y_pred = (y_scores >= threshold).astype(int)

    # Compute classification metrics
    classification_metrics = compute_classification_metrics(y_true, y_pred)

    logger.info(
        "Classification metrics - Precision: %.4f, Recall: %.4f, F1: %.4f",
        classification_metrics['precision'],
        classification_metrics['recall'],
        classification_metrics['f1']
    )

    results: Dict[str, Union[float, Dict]] = {
        'auroc': auroc,
        'average_precision': ap,
        'threshold': float(threshold),
        'metrics': classification_metrics,
    }

    if pixel_labels is not None and pixel_scores is not None:
        results['pixel_metrics'] = {
            'pixel_auroc': compute_pixel_auroc(pixel_labels, pixel_scores),
            'pixel_average_precision': compute_pixel_average_precision(pixel_labels, pixel_scores),
            'aupro': compute_aupro(
                pixel_labels,
                pixel_scores,
                integration_limit=pro_integration_limit,
            ),
        }

    return results


def compute_pro_score(
    pixel_labels: NDArray,
    pixel_scores: NDArray,
    integration_limit: float = 1.0,
) -> float:
    """
    Compute Per-Region-Overlap (PRO) score for pixel-level evaluation.

    PRO score is designed for anomaly localization evaluation.

    Parameters
    ----------
    pixel_labels : ndarray of shape (n_images, H, W)
        Ground truth pixel-level labels
    pixel_scores : ndarray of shape (n_images, H, W)
        Predicted anomaly scores for each pixel
    integration_limit : float, default=1.0
        FPR integration limit for PRO score

    Returns
    -------
    pro_score : float
        Per-Region-Overlap score

    Notes
    -----
    Higher PRO scores indicate better localization performance.

    References
    ----------
    Bergmann et al. (2020). "The MVTec Anomaly Detection Dataset"
    """
    if not (0.0 < float(integration_limit) <= 1.0):
        raise ValueError(f"integration_limit must be in (0, 1]. Got {integration_limit}.")

    pixel_labels_arr = np.asarray(pixel_labels)
    pixel_scores_arr = np.asarray(pixel_scores)
    if pixel_labels_arr.size == 0 or pixel_scores_arr.size == 0:
        raise ValueError("pixel_labels and pixel_scores must be non-empty.")
    if pixel_labels_arr.shape != pixel_scores_arr.shape:
        raise ValueError(
            "pixel_labels and pixel_scores must have the same shape. "
            f"Got {pixel_labels_arr.shape} != {pixel_scores_arr.shape}."
        )

    # Flatten arrays
    pixel_labels_flat = pixel_labels_arr.ravel()
    pixel_scores_flat = pixel_scores_arr.ravel()

    if len(np.unique(pixel_labels_flat)) < 2:
        logger.warning(
            "Only one class present in pixel_labels. PRO score is not defined."
        )
        return float("nan")

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(pixel_labels_flat, pixel_scores_flat)

    # Integrate TPR up to integration_limit FPR.
    # Use interpolation to ensure we include a point exactly at integration_limit.
    mask = fpr <= integration_limit
    fpr_clip = fpr[mask]
    tpr_clip = tpr[mask]

    if fpr_clip.size == 0:
        return 0.0

    if float(fpr_clip[-1]) < float(integration_limit):
        unique_fpr, inverse = np.unique(fpr, return_inverse=True)
        unique_tpr = np.full_like(unique_fpr, -np.inf, dtype=np.float64)
        np.maximum.at(unique_tpr, inverse, tpr.astype(np.float64))
        tpr_at_limit = float(np.interp(float(integration_limit), unique_fpr, unique_tpr))
        fpr_clip = np.append(fpr_clip, float(integration_limit))
        tpr_clip = np.append(tpr_clip, tpr_at_limit)

    # NumPy 2.x removed `np.trapz` in favor of `np.trapezoid`.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # pragma: no cover
        trapezoid = getattr(np, "trapz")

    area = float(trapezoid(tpr_clip, fpr_clip))
    return float(area / float(integration_limit))


def compute_pixel_auroc(
    pixel_labels: NDArray,
    pixel_scores: NDArray,
) -> float:
    """Compute pixel-level AUROC by flattening pixel arrays."""

    return compute_auroc(pixel_labels.ravel(), pixel_scores.ravel())


def compute_pixel_average_precision(
    pixel_labels: NDArray,
    pixel_scores: NDArray,
) -> float:
    """Compute pixel-level Average Precision (AP) by flattening pixel arrays."""

    return compute_average_precision(pixel_labels.ravel(), pixel_scores.ravel())


def compute_aupro(
    pixel_labels: NDArray,
    pixel_scores: NDArray,
    integration_limit: float = 0.3,
) -> float:
    """Alias for :func:`compute_pro_score` (commonly referred to as AUPRO)."""

    return compute_pro_score(
        pixel_labels=pixel_labels,
        pixel_scores=pixel_scores,
        integration_limit=integration_limit,
    )


def print_evaluation_summary(results: Dict) -> None:
    """
    Print formatted evaluation summary.

    Parameters
    ----------
    results : dict
        Results dictionary from evaluate_detector()

    Examples
    --------
    >>> results = evaluate_detector(y_true, y_scores)
    >>> print_evaluation_summary(results)

    ========================================
    Anomaly Detection Evaluation Summary
    ========================================

    Score-based Metrics:
    --------------------
    AUROC:               0.9523
    Average Precision:   0.9145

    Classification Metrics (threshold=0.523):
    ------------------------------------------
    Precision:           0.8750
    Recall (TPR):        0.9333
    F1-Score:            0.9032
    Specificity (TNR):   0.9500
    Accuracy:            0.9400

    Confusion Matrix:
    -----------------
    TP: 28    FP: 4
    FN: 2     TN: 66
    """
    print("\n" + "=" * 40)
    print("Anomaly Detection Evaluation Summary")
    print("=" * 40 + "\n")

    print("Score-based Metrics:")
    print("-" * 20)
    print(f"AUROC:               {results['auroc']:.4f}")
    print(f"Average Precision:   {results['average_precision']:.4f}")

    print(f"\nClassification Metrics (threshold={results['threshold']:.3f}):")
    print("-" * 42)
    metrics = results['metrics']
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall (TPR):        {metrics['recall']:.4f}")
    print(f"F1-Score:            {metrics['f1']:.4f}")
    print(f"Specificity (TNR):   {metrics['specificity']:.4f}")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")

    print("\nConfusion Matrix:")
    print("-" * 17)
    print(f"TP: {metrics['tp']:<5} FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']:<5} TN: {metrics['tn']}")
    print()
