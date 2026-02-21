"""
Visualization utilities for anomaly detection results.

This module provides functions to visualize:
- Anomaly heatmaps overlaid on images
- ROC curves and Precision-Recall curves
- Score distributions
- Comparison of multiple detectors
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    Figure = None

logger = logging.getLogger(__name__)


def plot_anomaly_map(
    image_path: str,
    anomaly_map: NDArray,
    threshold: Optional[float] = None,
    alpha: float = 0.4,
    cmap: str = 'jet',
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Visualize anomaly heatmap overlaid on original image.

    Parameters
    ----------
    image_path : str
        Path to original image
    anomaly_map : ndarray of shape (H, W)
        Anomaly heatmap (higher values = more anomalous)
    threshold : float, optional
        Threshold for binary mask overlay
    alpha : float, default=0.4
        Transparency of heatmap overlay (0=transparent, 1=opaque)
    cmap : str, default='jet'
        Colormap for heatmap ('jet', 'hot', 'viridis', etc.)
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if matplotlib available, else None

    Examples
    --------
    >>> from pyimgano.models import create_model
    >>> from pyimgano.visualization import plot_anomaly_map
    >>>
    >>> detector = create_model('vision_patchcore')
    >>> detector.fit(train_images)
    >>> anomaly_map = detector.get_anomaly_map('test.jpg')
    >>> plot_anomaly_map('test.jpg', anomaly_map)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is not available. Cannot create plots.")
        return None

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize anomaly map to match image size
    if anomaly_map.shape[:2] != img_rgb.shape[:2]:
        anomaly_map = cv2.resize(
            anomaly_map,
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

    # Normalize anomaly map to [0, 1]
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8
    )

    # Create figure
    if threshold is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Anomaly heatmap
    im = axes[1].imshow(anomaly_map_norm, cmap=cmap)
    axes[1].set_title('Anomaly Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    overlay = img_rgb.copy().astype(float) / 255.0
    heatmap_colored = plt.get_cmap(cmap)(anomaly_map_norm)[:, :, :3]
    overlay = (1 - alpha) * overlay + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (alpha={alpha})')
    axes[2].axis('off')

    # Binary mask (if threshold provided)
    if threshold is not None:
        binary_mask = anomaly_map_norm >= threshold
        axes[3].imshow(img_rgb)
        axes[3].imshow(
            binary_mask,
            cmap='Reds',
            alpha=0.5 * binary_mask.astype(float)
        )
        axes[3].set_title(f'Binary Mask (threshold={threshold:.2f})')
        axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Figure saved to %s", save_path)

    if show:
        plt.show()

    return fig


def plot_roc_curve(
    y_true: NDArray,
    y_scores: NDArray,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels (0 = normal, 1 = anomaly)
    y_scores : ndarray of shape (n_samples,)
        Anomaly scores
    label : str, optional
        Label for the curve
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.figure.Figure or None

    Examples
    --------
    >>> from pyimgano.visualization import plot_roc_curve
    >>> plot_roc_curve(y_true, y_scores, label='ECOD')
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is not available.")
        return None

    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    curve_label = f'{label} (AUC = {roc_auc:.3f})' if label else f'AUC = {roc_auc:.3f}'
    ax.plot(fpr, tpr, lw=2, label=curve_label)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("ROC curve saved to %s", save_path)

    if show:
        plt.show()

    return fig


def plot_precision_recall_curve(
    y_true: NDArray,
    y_scores: NDArray,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels
    y_scores : ndarray of shape (n_samples,)
        Anomaly scores
    label : str, optional
        Label for the curve
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is not available.")
        return None

    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))

    curve_label = f'{label} (AP = {ap:.3f})' if label else f'AP = {ap:.3f}'
    ax.plot(recall, precision, lw=2, label=curve_label)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("PR curve saved to %s", save_path)

    if show:
        plt.show()

    return fig


def plot_score_distribution(
    normal_scores: NDArray,
    anomaly_scores: NDArray,
    bins: int = 50,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot distribution of anomaly scores for normal and anomalous samples.

    Parameters
    ----------
    normal_scores : ndarray
        Scores for normal samples
    anomaly_scores : ndarray
        Scores for anomalous samples
    bins : int, default=50
        Number of histogram bins
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.figure.Figure or None

    Examples
    --------
    >>> normal_scores = detector.decision_function(normal_images)
    >>> anomaly_scores = detector.decision_function(anomaly_images)
    >>> plot_score_distribution(normal_scores, anomaly_scores)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is not available.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    ax.hist(
        normal_scores,
        bins=bins,
        alpha=0.6,
        label=f'Normal (n={len(normal_scores)})',
        color='blue',
        density=True
    )
    ax.hist(
        anomaly_scores,
        bins=bins,
        alpha=0.6,
        label=f'Anomaly (n={len(anomaly_scores)})',
        color='red',
        density=True
    )

    # Add vertical lines for means
    ax.axvline(
        np.mean(normal_scores),
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'Normal mean: {np.mean(normal_scores):.3f}'
    )
    ax.axvline(
        np.mean(anomaly_scores),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Anomaly mean: {np.mean(anomaly_scores):.3f}'
    )

    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Anomaly Score Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Score distribution saved to %s", save_path)

    if show:
        plt.show()

    return fig


def compare_detectors(
    y_true: NDArray,
    scores_dict: Dict[str, NDArray],
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Compare multiple anomaly detectors using ROC curves.

    Parameters
    ----------
    y_true : ndarray
        True binary labels
    scores_dict : dict
        Dictionary mapping detector names to their scores
    save_path : str, optional
        Path to save the figure
    show : bool, default=True
        Whether to display the figure

    Returns
    -------
    fig : matplotlib.figure.Figure or None

    Examples
    --------
    >>> scores_dict = {
    ...     'ECOD': ecod_detector.predict(test_images),
    ...     'PatchCore': patchcore_detector.predict(test_images),
    ...     'SimpleNet': simplenet_detector.predict(test_images),
    ... }
    >>> compare_detectors(test_labels, scores_dict)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is not available.")
        return None

    from sklearn.metrics import auc, roc_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot ROC curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(scores_dict)))

    for (name, scores), color in zip(scores_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC={roc_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve Comparison', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)

    # Bar chart of AUROC scores
    names = list(scores_dict.keys())
    aurocs = []
    for scores in scores_dict.values():
        from sklearn.metrics import roc_auc_score
        aurocs.append(roc_auc_score(y_true, scores))

    bars = ax2.bar(names, aurocs, color=colors[:len(names)])
    ax2.set_ylabel('AUROC', fontsize=12)
    ax2.set_title('AUROC Comparison', fontsize=14)
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, auroc in zip(bars, aurocs):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{auroc:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Rotate x-labels if many detectors
    if len(names) > 3:
        ax2.set_xticklabels(names, rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Comparison figure saved to %s", save_path)

    if show:
        plt.show()

    return fig


def save_anomaly_overlay(
    image_path: str,
    anomaly_map: NDArray,
    output_path: str,
    alpha: float = 0.5,
    cmap: str = 'jet',
) -> None:
    """
    Save anomaly heatmap overlaid on image (without matplotlib).

    Uses only OpenCV for faster processing and no display dependencies.

    Parameters
    ----------
    image_path : str
        Path to original image
    anomaly_map : ndarray of shape (H, W)
        Anomaly heatmap
    output_path : str
        Path to save output image
    alpha : float, default=0.5
        Overlay transparency
    cmap : str, default='jet'
        OpenCV colormap (COLORMAP_JET, etc.)

    Examples
    --------
    >>> save_anomaly_overlay('test.jpg', anomaly_map, 'test_overlay.jpg')
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize anomaly map
    if anomaly_map.shape[:2] != img.shape[:2]:
        anomaly_map = cv2.resize(
            anomaly_map,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

    # Normalize to [0, 255]
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8
    )
    anomaly_map_uint8 = (anomaly_map_norm * 255).astype(np.uint8)

    # Apply colormap
    colormap = getattr(cv2, f'COLORMAP_{cmap.upper()}', cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(anomaly_map_uint8, colormap)

    # Overlay
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    # Save
    cv2.imwrite(output_path, overlay)
    logger.info("Anomaly overlay saved to %s", output_path)
