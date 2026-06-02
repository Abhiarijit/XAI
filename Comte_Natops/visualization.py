"""
visualization.py — Publication-quality figures for CoMTE on NATOPS.

Figures generated:
    F1  — Training curves (loss + accuracy)
    F2  — Confusion matrix
    F3  — Class distribution (train vs test)
    F4  — CF time-series overlay (orig vs CF, per channel, up to 3 samples)
    F5  — Channel importance bar chart (swapped channels highlighted)
    F6  — Swap probability trace (P(y_tgt) per swap step)
    F7  — Delta heatmap (sensor × time)
    F8  — NUN comparison heatmap (X vs NUN vs X_cf)
    F9  — Metric bar chart (validity, proximity, sparsity, compactness)
    F10 — Proximity-Validity scatter
    F11 — Per-class validity breakdown
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path

from config import Config, cfg

SENSOR_SHORT = [
    "HR_x","HR_y","HR_z",
    "HL_x","HL_y","HL_z",
    "ER_x","ER_y","ER_z",
    "EL_x","EL_y","EL_z",
    "WR_x","WR_y","WR_z",
    "WL_x","WL_y","WL_z",
    "TR_x","TR_y","TR_z",
    "TL_x","TL_y","TL_z",
]


def _save(fig, name: str, c: Config) -> None:
    path = c.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] Saved → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, c: Config = cfg) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(history["val_loss"],   label="Val",   color="darkorange")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_acc"], color="seagreen")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy"); axes[1].grid(True, alpha=0.3)
    axes[1].axhline(max(history["val_acc"]), color="red", linestyle="--", alpha=0.5,
                    label=f"Best={max(history['val_acc']):.3f}")
    axes[1].legend()

    fig.suptitle("CoMTE-NATOPS — Transformer Training Curves", fontweight="bold")
    _save(fig, "F1_training_curves", c)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, c: Config = cfg) -> None:
    from sklearn.metrics import confusion_matrix
    K   = c.K
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    labels = [c.CLASS_NAMES[i] for i in range(K)]
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalised)", fontweight="bold")

    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=8)

    _save(fig, "F2_confusion_matrix", c)


def plot_class_distribution(y_tr: np.ndarray, y_te: np.ndarray, c: Config = cfg) -> None:
    K      = c.K
    labels = [c.CLASS_NAMES[i] for i in range(K)]
    x      = np.arange(K)
    w      = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, np.bincount(y_tr.astype(int), minlength=K), w, label="Train", color="steelblue")
    ax.bar(x + w/2, np.bincount(y_te.astype(int), minlength=K), w, label="Test",  color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Count"); ax.set_title("NATOPS Class Distribution", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "F3_class_distribution", c)


def plot_cf_timeseries(X_orig: np.ndarray,
                       X_cf:   np.ndarray,
                       info:   dict,
                       c:      Config = cfg,
                       sample_idx: int = 0) -> None:
    """
    Plot original vs CF time series for all 24 channels.
    Swapped channels are highlighted with a red background.
    """
    T, D      = X_orig.shape
    swapped   = set(info.get("swapped_channels", []))
    orig_cls  = info.get("orig_class", "?")
    cf_cls    = info.get("cf_class",   "?")
    valid     = info.get("validity",   False)

    n_cols = 6
    n_rows = math.ceil(D / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.2),
                              sharex=True)
    axes = axes.flatten()

    t = np.arange(T)
    for d in range(D):
        ax = axes[d]
        if d in swapped:
            ax.set_facecolor("#fff0f0")
        ax.plot(t, X_orig[:, d], color="steelblue", linewidth=1.2, label="Orig")
        ax.plot(t, X_cf[:,   d], color="firebrick",  linewidth=1.2, linestyle="--", label="CF")
        ax.set_title(SENSOR_SHORT[d], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if d in swapped:
            ax.set_title(f"{SENSOR_SHORT[d]} ✗", fontsize=8, color="firebrick")

    for d in range(D, len(axes)):
        axes[d].set_visible(False)

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"CoMTE CF #{sample_idx+1}  |  Orig:{c.CLASS_NAMES[orig_cls]} → CF:{c.CLASS_NAMES.get(cf_cls,'?')}  "
        f"[valid={'✓' if valid else '✗'}]  |  swapped={len(swapped)}/{D} channels",
        fontweight="bold", fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, f"F4_cf_timeseries_sample{sample_idx}", c)


def plot_channel_importance(info: dict, c: Config = cfg, sample_idx: int = 0) -> None:
    """Bar chart of channel importance (distance × swap indicator)."""
    import math
    D       = c.D
    w_imp   = info.get("channel_importance_weighted", np.zeros(D))
    swapped = set(info.get("swapped_channels", []))
    colors  = ["firebrick" if i in swapped else "steelblue" for i in range(D)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(D), w_imp, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(D))
    ax.set_xticklabels(SENSOR_SHORT, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Channel Distance (L2)")
    ax.set_title(
        f"CoMTE Channel Importance — Sample {sample_idx+1}  "
        f"(red = swapped, n={len(swapped)}/{D})",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="firebrick",  label="Swapped"),
        Patch(color="steelblue",  label="Unchanged"),
    ], fontsize=9)
    _save(fig, f"F5_channel_importance_sample{sample_idx}", c)


def plot_swap_trace(info: dict, c: Config = cfg, sample_idx: int = 0) -> None:
    """Line plot of P(y_tgt) as channels are progressively swapped."""
    trace   = info.get("prob_trace", [])
    if not trace:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(trace)), trace, marker="o", color="darkorange", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Decision threshold")
    ax.set_xlabel("Swap step (# channels swapped)")
    ax.set_ylabel("P(target class)")
    ax.set_title(
        f"CoMTE Probability Trace — Sample {sample_idx+1}  "
        f"[valid={info.get('validity', False)}]",
        fontweight="bold",
    )
    ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, f"F6_swap_trace_sample{sample_idx}", c)


def plot_delta_heatmap(X_orig: np.ndarray,
                       X_cf:   np.ndarray,
                       c:      Config = cfg,
                       sample_idx: int = 0) -> None:
    """Heatmap of |X_cf - X_orig| over (T, D)."""
    delta = np.abs(X_cf - X_orig).T   # (D, T)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(delta, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="|Δ|")
    ax.set_yticks(range(c.D))
    ax.set_yticklabels(SENSOR_SHORT, fontsize=8)
    ax.set_xlabel("Timestep")
    ax.set_title(f"CoMTE |Δ| Heatmap (Sensor × Time) — Sample {sample_idx+1}",
                 fontweight="bold")
    _save(fig, f"F7_delta_heatmap_sample{sample_idx}", c)


def plot_nun_comparison(X_orig: np.ndarray,
                        NUN:    np.ndarray,
                        X_cf:   np.ndarray,
                        c:      Config = cfg,
                        sample_idx: int = 0) -> None:
    """Side-by-side heatmaps: Original | NUN | CF."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Original X", "NUN (target-class neighbor)", "CF (after swaps)"]
    arrays = [X_orig.T, NUN.T, X_cf.T]   # each (D, T)

    vmin = min(a.min() for a in arrays)
    vmax = max(a.max() for a in arrays)

    for ax, arr, title in zip(axes, arrays, titles):
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(c.D))
        ax.set_yticklabels(SENSOR_SHORT, fontsize=7)
        ax.set_xlabel("Timestep")
        ax.set_title(title, fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"CoMTE: X vs NUN vs CF — Sample {sample_idx+1}", fontweight="bold")
    plt.tight_layout()
    _save(fig, f"F8_nun_comparison_sample{sample_idx}", c)


def plot_metric_comparison(agg: dict, c: Config = cfg) -> None:
    """Bar chart of aggregate CoMTE metrics."""
    keys   = ["validity", "proximity_l1", "proximity_l2", "sparsity", "compactness", "plausibility"]
    keys   = [k for k in keys if k in agg]
    means  = [agg[k]["mean"] for k in keys]
    stds   = [agg[k]["std"]  for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("CoMTE on NATOPS — Aggregate Metrics", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    _save(fig, "F9_metric_summary", c)


def plot_proximity_validity_scatter(results: list[dict], c: Config = cfg) -> None:
    """Scatter plot: L2 proximity vs validity (per-sample)."""
    prox = [r["proximity_l2"] for r in results]
    val  = [r["validity"]     for r in results]
    colors = ["green" if v else "red" for v in val]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(prox, val, c=colors, alpha=0.7, edgecolors="white", s=60)
    ax.set_xlabel("Proximity L2 (lower = closer to original)")
    ax.set_ylabel("Validity (1 = class flipped)")
    ax.set_title("CoMTE: Proximity vs Validity — NATOPS", fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Not valid", "Valid"])
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="green", label="Valid CF"),
        Patch(color="red",   label="Invalid CF"),
    ])
    _save(fig, "F10_proximity_validity_scatter", c)


def plot_per_class_validity(results: list[dict], c: Config = cfg) -> None:
    """Bar chart of validity rate per original class."""
    K = c.K
    class_valid = {k: [] for k in range(K)}
    for r in results:
        class_valid[r["orig_class"]].append(r["validity"])

    rates  = [np.mean(class_valid[k]) if class_valid[k] else 0.0 for k in range(K)]
    labels = [c.CLASS_NAMES[k] for k in range(K)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(K), rates, color="steelblue", edgecolor="white")
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Validity Rate")
    ax.set_title("CoMTE Validity Rate per Class — NATOPS", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(np.mean(rates), color="red", linestyle="--", alpha=0.6,
               label=f"Overall mean={np.mean(rates):.3f}")
    ax.legend()

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02,
                f"{rate:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, "F11_per_class_validity", c)


import math   # needed by plot_cf_timeseries
