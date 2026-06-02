"""
visualization.py — Publication-quality figures for AB-CF on NATOPS.

Figures:
    F1  — Training curves (loss + accuracy)
    F2  — Confusion matrix
    F3  — Class distribution (train vs test)
    F4  — CF time-series overlay (orig vs CF), swapped segments shaded
    F5  — Shannon entropy heatmap (channel × segment)
    F6  — Channel importance bar chart
    F7  — Segment importance bar chart
    F8  — Swap probability trace (P(y_tgt) per swap step)
    F9  — Delta heatmap (sensor × time)
    F10 — NUN comparison (X | NUN | CF heatmaps)
    F11 — Aggregate metric bar chart
    F12 — Proximity-Validity scatter
    F13 — Per-class validity breakdown
"""
from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

    fig.suptitle("AB-CF NATOPS — Transformer Training Curves", fontweight="bold")
    _save(fig, "F1_training_curves", c)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, c: Config = cfg) -> None:
    from sklearn.metrics import confusion_matrix
    K      = c.K
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    cm_n   = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    labels = [c.CLASS_NAMES[i] for i in range(K)]
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalised)", fontweight="bold")

    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_n[i,j] > 0.5 else "black", fontsize=8)
    _save(fig, "F2_confusion_matrix", c)


def plot_class_distribution(y_tr: np.ndarray, y_te: np.ndarray, c: Config = cfg) -> None:
    K      = c.K
    labels = [c.CLASS_NAMES[i] for i in range(K)]
    x, w   = np.arange(K), 0.35

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
    24-channel overlay plot. Swapped segments are shaded in red.
    """
    T, D     = X_orig.shape
    swapped  = info.get("swapped", [])
    segments = info.get("segments", [])
    orig_cls = info.get("orig_class", "?")
    cf_cls   = info.get("cf_class",   "?")
    valid    = info.get("validity",   False)

    # Build per-channel swapped segment set
    chan_segs: dict[int, list[tuple[int,int]]] = {}
    for d, s_idx, start, end, H in swapped:
        chan_segs.setdefault(d, []).append((start, end))

    n_cols = 6
    n_rows = math.ceil(D / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.2),
                              sharex=True)
    axes = axes.flatten()
    t    = np.arange(T)

    for d in range(D):
        ax = axes[d]
        ax.plot(t, X_orig[:, d], color="steelblue",  linewidth=1.2, label="Orig")
        ax.plot(t, X_cf[:,   d], color="firebrick",  linewidth=1.2, linestyle="--", label="CF")

        if d in chan_segs:
            for (start, end) in chan_segs[d]:
                ax.axvspan(start, end - 1, alpha=0.18, color="red")
            ax.set_title(f"{SENSOR_SHORT[d]} ✗", fontsize=8, color="firebrick")
        else:
            ax.set_title(SENSOR_SHORT[d], fontsize=8)

        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for d in range(D, len(axes)):
        axes[d].set_visible(False)

    axes[0].legend(fontsize=7, loc="upper right")
    n_swapped_chans = len(set(d for d, *_ in swapped))
    fig.suptitle(
        f"AB-CF CF #{sample_idx+1}  |  "
        f"Orig:{c.CLASS_NAMES[orig_cls]} → CF:{c.CLASS_NAMES.get(cf_cls,'?')}  "
        f"[valid={'✓' if valid else '✗'}]  |  "
        f"swaps={info.get('n_swaps',0)}, channels={n_swapped_chans}/{D}",
        fontweight="bold", fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, f"F4_cf_timeseries_sample{sample_idx}", c)


def plot_entropy_heatmap(info: dict, c: Config = cfg, sample_idx: int = 0) -> None:
    """Heatmap of Shannon entropy per (channel, segment)."""
    entropy_map = info.get("entropy_map", None)   # (D, n_seg)
    if entropy_map is None:
        return

    D, n_seg = entropy_map.shape
    segments = info.get("segments", [])
    seg_labels = [f"S{i}\n[{s},{e})" for i, (s, e) in enumerate(segments)]

    fig, ax = plt.subplots(figsize=(max(6, n_seg * 1.5), 8))
    im = ax.imshow(entropy_map, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Shannon Entropy (bits)")

    ax.set_yticks(range(D))
    ax.set_yticklabels(SENSOR_SHORT[:D], fontsize=8)
    ax.set_xticks(range(n_seg))
    ax.set_xticklabels(seg_labels, fontsize=8)
    ax.set_xlabel("Segment"); ax.set_ylabel("Channel")
    ax.set_title(f"AB-CF Shannon Entropy Map — Sample {sample_idx+1}",
                 fontweight="bold")

    # Mark swapped cells with an 'X'
    swapped = info.get("swapped", [])
    for d, s_idx, start, end, H in swapped:
        ax.text(s_idx, d, "×", ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")

    _save(fig, f"F5_entropy_heatmap_sample{sample_idx}", c)


def plot_channel_importance(info: dict, c: Config = cfg, sample_idx: int = 0) -> None:
    """Bar chart of per-channel importance (sum of swapped segment entropies)."""
    ch_imp  = info.get("channel_importance", np.zeros(c.D))
    swapped = set(d for d, *_ in info.get("swapped", []))
    colors  = ["firebrick" if i in swapped else "steelblue" for i in range(c.D)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(c.D), ch_imp, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(c.D))
    ax.set_xticklabels(SENSOR_SHORT, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative Entropy of Swapped Segments")
    ax.set_title(
        f"AB-CF Channel Importance — Sample {sample_idx+1}  (red = swapped)",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(handles=[
        Patch(color="firebrick", label="Swapped"),
        Patch(color="steelblue", label="Unchanged"),
    ], fontsize=9)
    _save(fig, f"F6_channel_importance_sample{sample_idx}", c)


def plot_segment_importance(info: dict, c: Config = cfg, sample_idx: int = 0) -> None:
    """Bar chart of segment importance (number of channels swapped per segment)."""
    seg_imp  = info.get("segment_importance", np.zeros(1))
    segments = info.get("segments", [])
    labels   = [f"S{i}\n[{s},{e})" for i, (s, e) in enumerate(segments)]

    fig, ax = plt.subplots(figsize=(max(6, len(segments) * 1.5), 4))
    colors = ["firebrick" if v > 0 else "steelblue" for v in seg_imp]
    ax.bar(range(len(segments)), seg_imp, color=colors, edgecolor="white")
    ax.set_xticks(range(len(segments)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("# Channels Swapped")
    ax.set_title(f"AB-CF Segment Importance — Sample {sample_idx+1}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, f"F7_segment_importance_sample{sample_idx}", c)


def plot_swap_trace(info: dict, c: Config = cfg, sample_idx: int = 0) -> None:
    """P(y_tgt) vs swap step."""
    trace = info.get("prob_trace", [])
    if not trace:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(trace)), trace, marker="o", color="darkorange", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="0.5 threshold")
    ax.set_xlabel("Swap step (# segments swapped)")
    ax.set_ylabel("P(target class)")
    ax.set_title(
        f"AB-CF Probability Trace — Sample {sample_idx+1}  "
        f"[valid={info.get('validity', False)}]",
        fontweight="bold",
    )
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, f"F8_swap_trace_sample{sample_idx}", c)


def plot_delta_heatmap(X_orig: np.ndarray,
                       X_cf:   np.ndarray,
                       info:   dict,
                       c:      Config = cfg,
                       sample_idx: int = 0) -> None:
    """Heatmap of |X_cf - X_orig| over (sensor, time) with segment boundaries."""
    delta    = np.abs(X_cf - X_orig).T   # (D, T)
    segments = info.get("segments", [])

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(delta, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="|Δ|")

    # Draw segment boundary lines
    for start, end in segments[1:]:
        ax.axvline(start - 0.5, color="cyan", linewidth=1.0, alpha=0.7)

    ax.set_yticks(range(c.D))
    ax.set_yticklabels(SENSOR_SHORT, fontsize=8)
    ax.set_xlabel("Timestep")
    ax.set_title(f"AB-CF |Δ| Heatmap (Sensor × Time) — Sample {sample_idx+1}",
                 fontweight="bold")
    _save(fig, f"F9_delta_heatmap_sample{sample_idx}", c)


def plot_nun_comparison(X_orig: np.ndarray,
                        NUN:    np.ndarray,
                        X_cf:   np.ndarray,
                        c:      Config = cfg,
                        sample_idx: int = 0) -> None:
    """Side-by-side heatmaps: Original | NUN | CF."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Original X", "NUN (target-class neighbor)", "AB-CF Result"]
    arrays = [X_orig.T, NUN.T, X_cf.T]

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

    fig.suptitle(f"AB-CF: X vs NUN vs CF — Sample {sample_idx+1}", fontweight="bold")
    plt.tight_layout()
    _save(fig, f"F10_nun_comparison_sample{sample_idx}", c)


def plot_metric_comparison(agg: dict, c: Config = cfg) -> None:
    """Aggregate metric bar chart."""
    keys  = ["validity", "proximity_l1", "proximity_l2", "sparsity", "compactness", "plausibility"]
    keys  = [k for k in keys if k in agg]
    means = [agg[k]["mean"] for k in keys]
    stds  = [agg[k]["std"]  for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color="darkorange", edgecolor="white", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("AB-CF on NATOPS — Aggregate Metrics", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
    _save(fig, "F11_metric_summary", c)


def plot_proximity_validity_scatter(results: list[dict], c: Config = cfg) -> None:
    """Proximity L2 vs validity scatter."""
    prox   = [r["proximity_l2"] for r in results]
    val    = [r["validity"]     for r in results]
    colors = ["green" if v else "red" for v in val]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(prox, val, c=colors, alpha=0.7, edgecolors="white", s=60)
    ax.set_xlabel("Proximity L2")
    ax.set_ylabel("Validity (1 = class flipped)")
    ax.set_title("AB-CF: Proximity vs Validity — NATOPS", fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Not valid", "Valid"])
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        Patch(color="green", label="Valid CF"),
        Patch(color="red",   label="Invalid CF"),
    ])
    _save(fig, "F12_proximity_validity_scatter", c)


def plot_per_class_validity(results: list[dict], c: Config = cfg) -> None:
    """Validity rate per original class."""
    K            = c.K
    class_valid  = {k: [] for k in range(K)}
    for r in results:
        class_valid[r["orig_class"]].append(r["validity"])

    rates  = [np.mean(class_valid[k]) if class_valid[k] else 0.0 for k in range(K)]
    labels = [c.CLASS_NAMES[k] for k in range(K)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(K), rates, color="darkorange", edgecolor="white")
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Validity Rate")
    ax.set_title("AB-CF Validity Rate per Class — NATOPS", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(np.mean(rates), color="red", linestyle="--", alpha=0.6,
               label=f"Overall mean={np.mean(rates):.3f}")
    ax.legend()

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02,
                f"{rate:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, "F13_per_class_validity", c)
