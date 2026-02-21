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
from scipy import ndimage
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
    pro_num_thresholds: int = 200,
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
                num_thresholds=pro_num_thresholds,
            ),
        }

    return results


def compute_pro_score(
    pixel_labels: NDArray,
    pixel_scores: NDArray,
    integration_limit: float = 1.0,
    num_thresholds: int = 200,
) -> float:
    """
    Compute Per-Region-Overlap (PRO) / AUPRO score for pixel-level evaluation.

    This implementation follows the common **AUPRO** definition used in
    industrial anomaly detection benchmarks (e.g., MVTec AD):

    1) Split the ground-truth anomaly mask into connected components (regions).
    2) For a set of thresholds, compute:
       - FPR: false positive rate on background pixels (gt==0)
       - PRO(thr): mean per-region overlap, i.e. mean(region_recall)
    3) Integrate PRO over FPR up to ``integration_limit`` and normalize by
       ``integration_limit``.

    Parameters
    ----------
    pixel_labels : ndarray of shape (n_images, H, W)
        Ground truth pixel-level labels
    pixel_scores : ndarray of shape (n_images, H, W)
        Predicted anomaly scores for each pixel
    integration_limit : float, default=1.0
        FPR integration limit for PRO score
    num_thresholds : int, default=200
        Number of thresholds sampled to approximate the AUPRO integral.
        Higher values are more accurate but slower.

    Returns
    -------
    pro_score : float
        Normalized area under the PRO curve (AUPRO) in [0, 1].

    Notes
    -----
    Higher PRO scores indicate better localization performance.

    References
    ----------
    Bergmann et al. (2020). "The MVTec Anomaly Detection Dataset"
    """
    if not (0.0 < float(integration_limit) <= 1.0):
        raise ValueError(f"integration_limit must be in (0, 1]. Got {integration_limit}.")
    if int(num_thresholds) < 10:
        raise ValueError(f"num_thresholds must be >= 10. Got {num_thresholds}.")

    pixel_labels_arr = np.asarray(pixel_labels)
    pixel_scores_arr = np.asarray(pixel_scores)
    if pixel_labels_arr.size == 0 or pixel_scores_arr.size == 0:
        raise ValueError("pixel_labels and pixel_scores must be non-empty.")
    if pixel_labels_arr.shape != pixel_scores_arr.shape:
        raise ValueError(
            "pixel_labels and pixel_scores must have the same shape. "
            f"Got {pixel_labels_arr.shape} != {pixel_scores_arr.shape}."
        )

    labels_bin = (pixel_labels_arr > 0).astype(np.uint8, copy=False)
    scores = np.asarray(pixel_scores_arr, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        # Avoid surprising ValueError in downstream computations; NaNs/inf would
        # otherwise propagate and make the metric meaningless.
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    if len(np.unique(labels_bin.ravel())) < 2:
        logger.warning(
            "Only one class present in pixel_labels. PRO score is not defined."
        )
        return float("nan")

    background_scores = scores[labels_bin == 0].ravel()
    if background_scores.size == 0:
        logger.warning("No background pixels found in pixel_labels. PRO score is not defined.")
        return float("nan")

    # Collect per-region score vectors (sorted) across the batch.
    region_scores_sorted: list[np.ndarray] = []
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity
    for i in range(int(labels_bin.shape[0])):
        cc_map, num_regions = ndimage.label(labels_bin[i].astype(bool), structure=structure)
        for region_id in range(1, int(num_regions) + 1):
            region_scores = scores[i][cc_map == region_id].ravel()
            if region_scores.size == 0:
                continue
            region_scores_sorted.append(np.sort(region_scores.astype(np.float64, copy=False)))

    if not region_scores_sorted:
        logger.warning("No connected-components regions found in pixel_labels. PRO score is not defined.")
        return float("nan")

    bg_sorted = np.sort(background_scores.astype(np.float64, copy=False))
    bg_n = int(bg_sorted.size)

    # Sample thresholds based on background quantiles so we spend resolution where
    # the PRO metric is typically integrated (low FPR).
    fpr_targets = np.linspace(0.0, float(integration_limit), int(num_thresholds), endpoint=True)
    thresholds = np.empty_like(fpr_targets, dtype=np.float64)

    # Ensure the first point yields exactly FPR=0: choose a threshold above max background.
    max_bg = float(bg_sorted[-1])
    thresholds[0] = max_bg + 1e-12
    if thresholds.size > 1:
        quantiles = 1.0 - fpr_targets[1:]
        try:
            thresholds[1:] = np.quantile(bg_sorted, quantiles, method="higher")
        except TypeError:  # pragma: no cover - older NumPy (<1.22)
            thresholds[1:] = np.quantile(bg_sorted, quantiles, interpolation="higher")

    # Compute actual FPR values (may differ slightly due to ties/quantile rounding).
    bg_left = np.searchsorted(bg_sorted, thresholds, side="left")
    fpr = (bg_n - bg_left) / float(bg_n)

    # Compute mean per-region overlap at each threshold.
    pro = np.zeros_like(fpr, dtype=np.float64)
    for t_idx, thr in enumerate(thresholds):
        overlaps = np.zeros(len(region_scores_sorted), dtype=np.float64)
        for r_idx, region_sorted in enumerate(region_scores_sorted):
            left = np.searchsorted(region_sorted, thr, side="left")
            overlaps[r_idx] = (region_sorted.size - left) / float(region_sorted.size)
        pro[t_idx] = float(np.mean(overlaps)) if overlaps.size else 0.0

    # Monotonicize/merge duplicates on the x-axis (keep best PRO per FPR).
    order = np.argsort(fpr)
    fpr_sorted = np.asarray(fpr[order], dtype=np.float64)
    pro_sorted = np.asarray(pro[order], dtype=np.float64)

    unique_fpr, inverse = np.unique(fpr_sorted, return_inverse=True)
    unique_pro = np.full_like(unique_fpr, -np.inf, dtype=np.float64)
    np.maximum.at(unique_pro, inverse, pro_sorted)

    # Ensure the curve starts at 0 FPR with 0 overlap (no positives predicted).
    if unique_fpr.size == 0:
        return 0.0
    if float(unique_fpr[0]) > 0.0:
        unique_fpr = np.insert(unique_fpr, 0, 0.0)
        unique_pro = np.insert(unique_pro, 0, 0.0)

    # Clip/integrate to the requested FPR limit, interpolating the last point.
    limit = float(integration_limit)
    mask = unique_fpr <= limit
    fpr_clip = unique_fpr[mask]
    pro_clip = unique_pro[mask]
    if fpr_clip.size == 0:
        return 0.0

    if float(fpr_clip[-1]) < limit:
        pro_at_limit = float(np.interp(limit, unique_fpr, unique_pro))
        fpr_clip = np.append(fpr_clip, limit)
        pro_clip = np.append(pro_clip, pro_at_limit)

    # NumPy 2.x removed `np.trapz` in favor of `np.trapezoid`.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # pragma: no cover
        trapezoid = getattr(np, "trapz")

    area = float(trapezoid(pro_clip, fpr_clip))
    return float(area / limit)


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
    num_thresholds: int = 200,
) -> float:
    """Alias for :func:`compute_pro_score` (commonly referred to as AUPRO)."""

    return compute_pro_score(
        pixel_labels=pixel_labels,
        pixel_scores=pixel_scores,
        integration_limit=integration_limit,
        num_thresholds=num_thresholds,
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
