"""
visualization.py — Publication figures for TAGFC_CMAPSS.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import Config, cfg

CLASS_NAMES = ["Healthy", "Degrading", "Critical"]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {path}")


def plot_training_curves(history: dict, c: Config = cfg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("TransformerCMAPSS — Training", fontweight="bold")

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    _save(fig, c.FIG_DIR / "F1_training_curves.png")


def plot_confusion_matrix(y_true, y_pred, c: Config = cfg):
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — TransformerCMAPSS (3-class)")
    _save(fig, c.FIG_DIR / "F2_confusion_matrix.png")


def plot_cf_timeseries(X_orig, X_cf, orig_cls, cf_cls,
                       sensor_names, sample_idx: int, c: Config = cfg):
    T, D = X_orig.shape
    ncols = 4
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    fig.suptitle(
        f"Sample {sample_idx} | Original: {CLASS_NAMES[orig_cls]} → CF: {CLASS_NAMES[cf_cls]}",
        fontweight="bold",
    )
    delta = X_cf - X_orig
    for d in range(D):
        ax = axes[d]
        ax.plot(X_orig[:, d], "b-",  label="Original",      linewidth=1.5)
        ax.plot(X_cf[:, d],   "r--", label="Counterfactual", linewidth=1.5)
        sig = np.abs(delta[:, d]) > 0.05
        if sig.any():
            ax.fill_between(range(T), X_orig[:, d], X_cf[:, d],
                            where=sig, alpha=0.25, color="red")
        ax.set_title(sensor_names[d], fontsize=8)
        ax.set_xlabel("Cycle")
        ax.tick_params(labelsize=7)
    for d in range(D, len(axes)):
        axes[d].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, c.FIG_DIR / f"F3_cf_timeseries_sample{sample_idx}.png")


def plot_temporal_saliency(saliency: np.ndarray, sample_idx: int, c: Config = cfg):
    T = len(saliency)
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = plt.cm.RdYlGn_r(saliency)
    for t in range(T):
        ax.bar(t, saliency[t], color=colors[t], width=0.9)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Saliency")
    ax.set_xlabel("Cycle (timestep)")
    ax.set_ylabel("Attention Weight")
    ax.set_title(f"Temporal Saliency (Attention Rollout) — Sample {sample_idx}")
    _save(fig, c.FIG_DIR / f"F4_temporal_saliency_sample{sample_idx}.png")


def plot_metric_comparison(agg_t, agg_c, agg_a, c: Config = cfg):
    metrics     = ["validity", "proximity_l1", "sparsity", "cf_confidence", "rcf"]
    labels      = ["Validity↑", "Prox L1↓", "Sparsity↑", "CF Conf↑", "RCF↓"]
    methods     = {"TAGFC": agg_t, "CoMTE": agg_c, "AB-CF": agg_a}
    colors      = {"TAGFC": "#4C72B0", "CoMTE": "#DD8452", "AB-CF": "#8E72B0"}

    x   = np.arange(len(metrics))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, agg) in enumerate(methods.items()):
        means = [agg.get(m, {}).get("mean", 0) for m in metrics]
        stds  = [agg.get(m, {}).get("std",  0) for m in metrics]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds,
               label=name, color=colors[name], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("TAGFC vs CoMTE vs AB-CF — CMAPSS FD001", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, c.FIG_DIR / "F5_metric_comparison.png")


def plot_proximity_validity_scatter(results_dict: dict, c: Config = cfg):
    colors = {"TAGFC": "#4C72B0", "CoMTE": "#DD8452", "AB-CF": "#8E72B0"}
    fig, ax = plt.subplots(figsize=(9, 4))
    rng = np.random.default_rng(0)
    for name, results in results_dict.items():
        l2s   = [r["proximity_l2"] for r in results]
        vals  = [r["validity"]     for r in results]
        jitter = rng.uniform(-0.02, 0.02, len(vals))
        ax.scatter(l2s, np.array(vals, float) + jitter,
                   alpha=0.6, s=30, label=name, color=colors.get(name, "gray"))
    ax.set_xlabel("L2 Distance to Original (lower = better)")
    ax.set_ylabel("Validity (1 = class flipped)")
    ax.set_title("Proximity vs Validity — CMAPSS FD001", fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Invalid (0)", "Valid (1)"])
    ax.legend()
    _save(fig, c.FIG_DIR / "F6_proximity_validity_scatter.png")
