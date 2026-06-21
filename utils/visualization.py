"""
Visualisation utilities for DG-HMCF.

All functions save figures to disk and optionally return the figure object
for notebook display.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import seaborn as sns
    _SNS_AVAILABLE = True
except ImportError:
    _SNS_AVAILABLE = False


MODALITY_NAMES = ["Speech", "Text", "Face", "EEG"]
MODALITY_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def _check_mpl():
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for visualisation functions.")


# ---------------------------------------------------------------------------
# Reliability weight bar chart
# ---------------------------------------------------------------------------

def plot_reliability_weights(
    weights: np.ndarray,
    modality_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Dynamic Modality Reliability Weights",
) -> Optional["plt.Figure"]:
    """
    Bar chart of mean reliability weights across a dataset.

    Parameters
    ----------
    weights : np.ndarray, shape (N, 4) or (4,)
        Per-sample or aggregate weights for each modality.
    modality_names : list of str, optional
        Labels for each modality.
    save_path : str, optional
        Path to save the figure (PNG).
    title : str
        Figure title.

    Returns
    -------
    matplotlib Figure or None
    """
    _check_mpl()
    if modality_names is None:
        modality_names = MODALITY_NAMES

    if weights.ndim == 2:
        mean_weights = weights.mean(axis=0)
        std_weights = weights.std(axis=0)
    else:
        mean_weights = weights
        std_weights = None

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        modality_names,
        mean_weights,
        color=MODALITY_COLORS,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
    if std_weights is not None:
        ax.errorbar(
            modality_names, mean_weights, yerr=std_weights,
            fmt="none", color="black", capsize=5, linewidth=1.5
        )

    ax.set_xlabel("Modality", fontsize=13)
    ax.set_ylabel("Mean Reliability Weight", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.0, mean_weights.max() * 1.4))

    # Annotate bars
    for bar, w in zip(bars, mean_weights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{w:.3f}",
            ha="center", va="bottom", fontsize=11,
        )

    ax.grid(axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attention_weights: Union[np.ndarray, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    title: str = "Cross-Modal Attention Weights",
) -> Optional["plt.Figure"]:
    """
    Visualise cross-modal attention matrices as heatmaps.

    Parameters
    ----------
    attention_weights : np.ndarray (T_q, T_kv) or dict of pair → array
    save_path : str, optional
    title : str

    Returns
    -------
    matplotlib Figure or None
    """
    _check_mpl()

    if isinstance(attention_weights, dict):
        n_maps = len(attention_weights)
        if n_maps == 0:
            return None

        cols = min(3, n_maps)
        rows = (n_maps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_maps == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        for ax, (pair_name, attn) in zip(axes, attention_weights.items()):
            _draw_heatmap(ax, np.array(attn).squeeze(), title=pair_name.replace("_", " ↔ "))

        # Hide unused axes
        for ax in axes[n_maps:]:
            ax.set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold")

    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        _draw_heatmap(ax, np.array(attention_weights).squeeze(), title=title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _draw_heatmap(ax, data: np.ndarray, title: str = "") -> None:
    if data.ndim == 0:
        data = data.reshape(1, 1)
    elif data.ndim == 1:
        data = data.reshape(1, -1)

    if _SNS_AVAILABLE:
        sns.heatmap(
            data, ax=ax, cmap="Blues", annot=data.size <= 100,
            fmt=".2f", linewidths=0.5, cbar=True,
        )
    else:
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax)

    ax.set_title(title, fontsize=11)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> Optional["plt.Figure"]:
    """
    Plot training and validation loss + F1 curves.

    Parameters
    ----------
    history  : dict with keys like 'train_loss', 'val_loss', 'val_f1', 'val_auc'
    save_path: str, optional
    title    : str

    Returns
    -------
    matplotlib Figure or None
    """
    _check_mpl()

    n_epochs = max((len(v) for v in history.values()), default=0)
    if n_epochs == 0:
        return None

    epochs = list(range(1, n_epochs + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Loss plot -------------------------------------------------------
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(epochs[:len(history["train_loss"])],
                history["train_loss"], label="Train Loss",
                color="#4C72B0", linewidth=2)
    if "val_loss" in history:
        ax.plot(epochs[:len(history["val_loss"])],
                history["val_loss"], label="Val Loss",
                color="#C44E52", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Metric plot -----------------------------------------------------
    ax = axes[1]
    metric_keys = {
        "val_f1": ("F1", "#55A868", "-"),
        "val_auc": ("AUROC", "#8172B2", "--"),
        "val_accuracy": ("Accuracy", "#C44E52", "-."),
    }
    for key, (label, color, style) in metric_keys.items():
        if key in history and history[key]:
            ax.plot(epochs[:len(history[key])],
                    history[key], label=label,
                    color=color, linewidth=2, linestyle=style)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Validation Metrics", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot a normalised or raw confusion matrix.

    Parameters
    ----------
    labels      : ground-truth int labels
    preds       : predicted int labels
    class_names : list of class name strings
    save_path   : str, optional
    normalize   : if True, show row-normalised proportions
    """
    _check_mpl()
    if class_names is None:
        class_names = ["Non-Depressed", "Depressed"]

    # Compute confusion matrix manually
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels.flatten(), preds.flatten()):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[int(t), int(p)] += 1

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        fmt = ".2%"
    else:
        cm_display = cm.astype(float)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    if _SNS_AVAILABLE:
        sns.heatmap(
            cm_display, ax=ax, annot=True, fmt=fmt,
            cmap="Blues", xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, cbar=True,
        )
    else:
        im = ax.imshow(cm_display, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f"{cm_display[i, j]:.2f}" if normalize else str(int(cm_display[i, j])),
                        ha="center", va="center", fontsize=11)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Modality importance
# ---------------------------------------------------------------------------

def plot_modality_importance(
    importance_scores: np.ndarray,
    modality_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Modality Importance Scores",
    group_by_label: Optional[np.ndarray] = None,
) -> Optional["plt.Figure"]:
    """
    Visualise per-modality importance scores.

    Parameters
    ----------
    importance_scores : np.ndarray, shape (N, 4) or (4,)
    modality_names    : list of str
    save_path         : str, optional
    group_by_label    : np.ndarray of int labels (0/1) for grouped bar chart
    """
    _check_mpl()
    if modality_names is None:
        modality_names = MODALITY_NAMES

    if importance_scores.ndim == 1:
        importance_scores = importance_scores.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(9, 5))

    if group_by_label is not None and len(np.unique(group_by_label)) == 2:
        # Grouped bar chart: depressed vs non-depressed
        dep_mask = group_by_label == 1
        non_dep_mask = group_by_label == 0
        mean_dep = importance_scores[dep_mask].mean(axis=0) if dep_mask.any() else np.zeros(4)
        mean_nondep = importance_scores[non_dep_mask].mean(axis=0) if non_dep_mask.any() else np.zeros(4)

        x = np.arange(len(modality_names))
        width = 0.35
        ax.bar(x - width / 2, mean_nondep, width, label="Non-Depressed",
               color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.bar(x + width / 2, mean_dep, width, label="Depressed",
               color="#C44E52", alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(modality_names)
        ax.legend(fontsize=11)
    else:
        mean_scores = importance_scores.mean(axis=0)
        bars = ax.bar(modality_names, mean_scores, color=MODALITY_COLORS,
                      alpha=0.85, edgecolor="black", linewidth=0.8)
        for bar, score in zip(bars, mean_scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{score:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_xlabel("Modality", fontsize=13)
    ax.set_ylabel("Mean Importance Score", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# PHQ-8 regression scatter plot
# ---------------------------------------------------------------------------

def plot_phq8_scatter(
    true_scores: np.ndarray,
    pred_scores: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "PHQ-8 Score Prediction",
) -> Optional["plt.Figure"]:
    """Scatter plot of true vs predicted PHQ-8 scores with regression line."""
    _check_mpl()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_scores, pred_scores, alpha=0.5, color="#4C72B0", s=30, edgecolors="none")

    # Regression line
    z = np.polyfit(true_scores, pred_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(true_scores.min(), true_scores.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Regression line")

    # Identity line
    lim_min = min(true_scores.min(), pred_scores.min())
    lim_max = max(true_scores.max(), pred_scores.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.5,
            linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("True PHQ-8 Score", fontsize=13)
    ax.set_ylabel("Predicted PHQ-8 Score", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
