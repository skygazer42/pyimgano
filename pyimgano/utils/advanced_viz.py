"""
Advanced visualization utilities for anomaly detection.

Provides comprehensive visualization tools:
- ROC curves, PR curves
- Confusion matrices
- Feature space visualization (t-SNE, UMAP)
- Anomaly heatmaps
- Score distributions
- Multi-model comparisons

Example:
    >>> from pyimgano.utils.advanced_viz import plot_roc_curve, plot_confusion_matrix
    >>> plot_roc_curve(y_true, y_scores)
    >>> plot_confusion_matrix(y_true, y_pred)
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_roc_curve(
    y_true: NDArray,
    y_scores: NDArray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[float, plt.Figure]:
    """Plot ROC curve.

    Args:
        y_true: Ground truth labels [N]
        y_scores: Predicted scores [N]
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        auc_score: AUC-ROC score
        fig: matplotlib figure

    Example:
        >>> auc_score, fig = plot_roc_curve(y_true, y_scores)
        >>> print(f"AUC-ROC: {auc_score:.4f}")
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return roc_auc, fig


def plot_pr_curve(
    y_true: NDArray,
    y_scores: NDArray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[float, plt.Figure]:
    """Plot Precision-Recall curve.

    Args:
        y_true: Ground truth labels [N]
        y_scores: Predicted scores [N]
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        ap_score: Average Precision score
        fig: matplotlib figure

    Example:
        >>> ap_score, fig = plot_pr_curve(y_true, y_scores)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AP = {ap_score:.4f})")
    ax.axhline(
        y=y_true.mean(),
        color="red",
        linestyle="--",
        lw=2,
        label=f"Baseline (AP = {y_true.mean():.4f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return ap_score, fig


def plot_confusion_matrix(
    y_true: NDArray,
    y_pred: NDArray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]
        labels: Class labels
        title: Plot title
        normalize: Whether to normalize values
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure

    Example:
        >>> plot_confusion_matrix(y_true, y_pred, labels=['Normal', 'Anomaly'])
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if labels is None:
        labels = ["Normal", "Anomaly"]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        square=True,
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 14},
    )

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_score_distribution(
    normal_scores: NDArray,
    anomaly_scores: NDArray,
    title: str = "Score Distribution",
    bins: int = 50,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot score distributions for normal and anomaly samples.

    Args:
        normal_scores: Scores for normal samples [N]
        anomaly_scores: Scores for anomaly samples [M]
        title: Plot title
        bins: Number of histogram bins
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure

    Example:
        >>> plot_score_distribution(normal_scores, anomaly_scores)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        normal_scores,
        bins=bins,
        alpha=0.6,
        color="blue",
        label=f"Normal (n={len(normal_scores)})",
        density=True,
    )
    ax.hist(
        anomaly_scores,
        bins=bins,
        alpha=0.6,
        color="red",
        label=f"Anomaly (n={len(anomaly_scores)})",
        density=True,
    )

    ax.axvline(
        normal_scores.mean(),
        color="blue",
        linestyle="--",
        lw=2,
        label=f"Normal mean: {normal_scores.mean():.3f}",
    )
    ax.axvline(
        anomaly_scores.mean(),
        color="red",
        linestyle="--",
        lw=2,
        label=f"Anomaly mean: {anomaly_scores.mean():.3f}",
    )

    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_feature_space_tsne(
    features: NDArray,
    labels: NDArray,
    title: str = "t-SNE Feature Space",
    perplexity: int = 30,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Visualize feature space using t-SNE.

    Args:
        features: Feature vectors [N, D]
        labels: Labels [N]
        title: Plot title
        perplexity: t-SNE perplexity parameter
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure

    Example:
        >>> plot_feature_space_tsne(features, labels)
    """
    from sklearn.manifold import TSNE

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot normal samples
    normal_mask = labels == 0
    ax.scatter(
        features_2d[normal_mask, 0],
        features_2d[normal_mask, 1],
        c="blue",
        label="Normal",
        alpha=0.6,
        s=50,
    )

    # Plot anomaly samples
    anomaly_mask = labels == 1
    ax.scatter(
        features_2d[anomaly_mask, 0],
        features_2d[anomaly_mask, 1],
        c="red",
        label="Anomaly",
        alpha=0.6,
        s=50,
        marker="x",
    )

    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_anomaly_heatmap(
    image: NDArray,
    anomaly_map: NDArray,
    title: str = "Anomaly Heatmap",
    alpha: float = 0.5,
    cmap: str = "jet",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot image with anomaly heatmap overlay.

    Args:
        image: Input image [H, W, C]
        anomaly_map: Anomaly map [H, W]
        title: Plot title
        alpha: Overlay alpha
        cmap: Colormap for heatmap
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure

    Example:
        >>> plot_anomaly_heatmap(image, anomaly_map)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Anomaly map
    im = axes[1].imshow(anomaly_map, cmap=cmap)
    axes[1].set_title("Anomaly Map", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(anomaly_map, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_multi_model_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Compare multiple models across metrics.

    Args:
        model_names: List of model names
        metrics: Dictionary of metric_name -> [values for each model]
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure

    Example:
        >>> metrics = {
        ...     'AUC-ROC': [0.95, 0.92, 0.98],
        ...     'AP': [0.90, 0.88, 0.96],
        ...     'F1': [0.85, 0.82, 0.92]
        ... }
        >>> plot_multi_model_comparison(['Model A', 'Model B', 'Model C'], metrics)
    """
    n_metrics = len(metrics)
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, model in enumerate(model_names):
        values = [metrics[metric][i] for metric in metrics.keys()]
        ax.bar(x + i * width, values, width, label=model, alpha=0.8)

    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics.keys())
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_threshold_analysis(
    y_true: NDArray,
    y_scores: NDArray,
    thresholds: Optional[NDArray] = None,
    title: str = "Threshold Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Analyze detection performance across different thresholds.

    Args:
        y_true: Ground truth labels [N]
        y_scores: Predicted scores [N]
        thresholds: Threshold values to evaluate
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        fig: matplotlib figure

    Example:
        >>> plot_threshold_analysis(y_true, y_scores)
    """
    if thresholds is None:
        thresholds = np.linspace(y_scores.min(), y_scores.max(), 100)

    from sklearn.metrics import f1_score, precision_score, recall_score

    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        if y_pred.sum() == 0:  # No positive predictions
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
        else:
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, precisions, label="Precision", lw=2)
    ax.plot(thresholds, recalls, label="Recall", lw=2)
    ax.plot(thresholds, f1_scores, label="F1 Score", lw=2)

    # Mark best F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    ax.axvline(
        best_threshold,
        color="red",
        linestyle="--",
        lw=2,
        label=f"Best threshold: {best_threshold:.3f} (F1={best_f1:.3f})",
    )

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_evaluation_report(
    y_true: NDArray,
    y_scores: NDArray,
    y_pred: NDArray,
    model_name: str = "Model",
    save_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """Create comprehensive evaluation report with multiple plots.

    Args:
        y_true: Ground truth labels [N]
        y_scores: Predicted scores [N]
        y_pred: Predicted labels [N]
        model_name: Name of the model
        save_dir: Directory to save figures

    Returns:
        Dictionary of figure_name -> matplotlib figure

    Example:
        >>> figures = create_evaluation_report(y_true, y_scores, y_pred, 'PatchCore')
    """
    figures = {}

    # ROC curve
    _, fig_roc = plot_roc_curve(
        y_true,
        y_scores,
        title=f"{model_name} - ROC Curve",
        save_path=f"{save_dir}/roc_curve.png" if save_dir else None,
        show=False,
    )
    figures["roc_curve"] = fig_roc

    # PR curve
    _, fig_pr = plot_pr_curve(
        y_true,
        y_scores,
        title=f"{model_name} - Precision-Recall Curve",
        save_path=f"{save_dir}/pr_curve.png" if save_dir else None,
        show=False,
    )
    figures["pr_curve"] = fig_pr

    # Confusion matrix
    fig_cm = plot_confusion_matrix(
        y_true,
        y_pred,
        title=f"{model_name} - Confusion Matrix",
        save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None,
        show=False,
    )
    figures["confusion_matrix"] = fig_cm

    # Score distribution
    normal_scores = y_scores[y_true == 0]
    anomaly_scores = y_scores[y_true == 1]

    fig_dist = plot_score_distribution(
        normal_scores,
        anomaly_scores,
        title=f"{model_name} - Score Distribution",
        save_path=f"{save_dir}/score_distribution.png" if save_dir else None,
        show=False,
    )
    figures["score_distribution"] = fig_dist

    # Threshold analysis
    fig_thresh = plot_threshold_analysis(
        y_true,
        y_scores,
        title=f"{model_name} - Threshold Analysis",
        save_path=f"{save_dir}/threshold_analysis.png" if save_dir else None,
        show=False,
    )
    figures["threshold_analysis"] = fig_thresh

    return figures
