"""
visualization.py — Publication-quality figures for the TAGFC paper.

Figures produced:
    F1  Training curves (loss + accuracy)
    F2  Confusion matrix
    F3  Original vs Counterfactual time series (all 14 sensors)
    F4  Temporal saliency (attention rollout)
    F5  Sensor importance bar chart
    F6  Frequency importance bar chart
    F7  2-D heatmap: sensor × frequency coefficient magnitude
    F8  Metric comparison bar chart (TAGFC vs CoMTE vs AB-CF)
    F9  Scatter: proximity_l2 vs validity (all methods)
    F10 Omega heatmap: per-coefficient penalty weight
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from config import Config, cfg

CLASS_NAMES  = {0: "Healthy", 1: "Degrading", 2: "Critical"}
CLASS_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
METHOD_COLORS = {"TAGFC": "#3498db", "CoMTE": "#e67e22", "AB-CF": "#9b59b6"}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":   150,
})


def _save(fig: plt.Figure, name: str, c: Config = cfg) -> None:
    path = c.FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [Fig] Saved → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# F1 — Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, c: Config = cfg) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", color="#3498db")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#e74c3c")
    ax1.set(xlabel="Epoch", ylabel="Cross-Entropy Loss", title="Training Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", color="#3498db")
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="#e74c3c")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Classification Accuracy",
            ylim=(0, 1.05))
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("TransformerFD001 — CMAPSS FD001 Training", fontweight="bold")
    fig.tight_layout()
    _save(fig, "F1_training_curves", c)


# ─────────────────────────────────────────────────────────────────────────────
# F2 — Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          c: Config = cfg) -> None:
    K    = c.K
    cm   = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ticks = [CLASS_NAMES[i] for i in range(K)]
    ax.set(xticks=range(K), yticks=range(K),
           xticklabels=ticks, yticklabels=ticks,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix — TransformerFD001")

    thresh = cm.max() / 2
    for i in range(K):
        for j in range(K):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold")

    fig.tight_layout()
    _save(fig, "F2_confusion_matrix", c)


# ─────────────────────────────────────────────────────────────────────────────
# F3 — Original vs Counterfactual time series
# ─────────────────────────────────────────────────────────────────────────────

def plot_cf_timeseries(X_orig: np.ndarray, X_cf: np.ndarray,
                       orig_class: int, cf_class: int,
                       sensor_names: tuple = cfg.SENSOR_NAMES,
                       sample_idx: int = 0,
                       c: Config = cfg) -> None:
    T, D = X_orig.shape
    cols = 4
    rows = (D + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.8), sharex=True)
    axes = axes.flatten()
    t    = np.arange(T)

    for d in range(D):
        ax = axes[d]
        ax.plot(t, X_orig[:, d], color="#2c3e50", lw=1.8, label="Original")
        ax.plot(t, X_cf[:, d],   color="#e74c3c",  lw=1.8, linestyle="--",
                label="Counterfactual")
        ax.fill_between(t, X_orig[:, d], X_cf[:, d], alpha=0.15, color="#e74c3c")
        ax.set_title(sensor_names[d], fontsize=10)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8)

    for ax in axes[D:]:
        ax.set_visible(False)

    title = (f"Sample {sample_idx}  |  "
             f"Original: {CLASS_NAMES[orig_class]}  →  "
             f"CF: {CLASS_NAMES[cf_class]}")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"F3_cf_timeseries_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F4 — Temporal saliency
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_saliency(saliency: np.ndarray, sample_idx: int = 0,
                           c: Config = cfg) -> None:
    T   = len(saliency)
    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = plt.cm.YlOrRd
    for t in range(T - 1):
        ax.fill_between([t, t + 1], 0, [saliency[t], saliency[t + 1]],
                        color=cmap(saliency[t]), alpha=0.85)
    ax.plot(np.arange(T), saliency, color="#2c3e50", lw=1.5)
    ax.set(xlabel="Cycle (relative)", ylabel="Attention Weight",
           title=f"Temporal Saliency (Attention Rollout) — Sample {sample_idx}",
           ylim=(0, 1.05))
    ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label="Saliency")
    fig.tight_layout()
    _save(fig, f"F4_temporal_saliency_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F5 — Sensor importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensor_importance(sensor_imp: np.ndarray,
                           sensor_names: tuple = cfg.SENSOR_NAMES,
                           sample_idx: int = 0,
                           c: Config = cfg) -> None:
    D    = len(sensor_imp)
    idx  = np.argsort(sensor_imp)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(D), sensor_imp[idx],
                  color=["#e74c3c" if sensor_imp[i] > sensor_imp.mean() else "#95a5a6"
                         for i in idx])
    ax.set(xticks=range(D),
           xticklabels=[sensor_names[i] for i in idx],
           ylabel="Σ|Δcoeff|",
           title=f"Sensor Importance (wavelet perturbation) — Sample {sample_idx}")
    ax.axhline(sensor_imp.mean(), color="#2c3e50", linestyle="--",
               alpha=0.6, label="Mean")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, f"F5_sensor_importance_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F6 — Frequency importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_freq_importance(freq_imp: np.ndarray, wav_levels: int = 3,
                         sample_idx: int = 0, c: Config = cfg) -> None:
    L   = len(freq_imp)
    n_a = L >> wav_levels                        # approximation band length

    bands, labels, colors = [], [], []
    bands.append((0, n_a, "Approx (slow trend)", "#e74c3c"))
    size = n_a
    for lvl in range(wav_levels):
        band_label = f"Detail-L{wav_levels - lvl}"
        col        = ["#f39c12", "#3498db", "#2ecc71"][lvl % 3]
        bands.append((n_a + lvl * size, n_a + (lvl + 1) * size, band_label, col))
        size *= 2

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(range(L), freq_imp, color="#bdc3c7", width=1.0, edgecolor="none")

    for start, end, label, col in bands:
        seg = freq_imp[start:end]
        ax.bar(range(start, start + len(seg)), seg, color=col,
               width=1.0, edgecolor="none", label=label, alpha=0.85)

    ax.set(xlabel="Wavelet Coefficient Index", ylabel="Σ|Δcoeff|",
           title=f"Frequency Importance (Haar DWT) — Sample {sample_idx}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, f"F6_freq_importance_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F7 — 2-D sensor × frequency heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_heatmap(delta: np.ndarray,
                       sensor_names: tuple = cfg.SENSOR_NAMES,
                       sample_idx: int = 0, c: Config = cfg) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = np.abs(delta).max() + 1e-8
    im   = ax.imshow(np.abs(delta), aspect="auto", cmap="hot", vmin=0, vmax=vmax)
    ax.set(yticks=range(len(sensor_names)),
           yticklabels=sensor_names,
           xlabel="Wavelet Coefficient Index",
           ylabel="Sensor",
           title=f"|Δ| Heatmap (Sensor × Frequency) — Sample {sample_idx}")
    plt.colorbar(im, ax=ax, label="|Δ coefficient|")
    fig.tight_layout()
    _save(fig, f"F7_delta_heatmap_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F8 — Metric comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_comparison(agg_tagfc:  dict,
                           agg_comte:  dict,
                           agg_abcf:   dict,
                           c: Config = cfg) -> None:
    metrics = ["validity", "proximity_l1", "sparsity", "cf_confidence", "rcf"]
    labels  = ["Validity ↑", "Prox L1 ↓", "Sparsity ↑", "CF Conf ↑", "RCF ↓"]
    methods = {"TAGFC": agg_tagfc, "CoMTE": agg_comte, "AB-CF": agg_abcf}
    n = len(metrics)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (name, agg) in enumerate(methods.items()):
        means = [agg.get(m, {}).get("mean", 0) for m in metrics]
        stds  = [agg.get(m, {}).get("std",  0) for m in metrics]
        ax.bar(x + (i - 1) * w, means, w,
               yerr=stds, capsize=4,
               color=METHOD_COLORS.get(name, "gray"),
               label=name, alpha=0.85, edgecolor="white")

    ax.set(xticks=x, xticklabels=labels,
           ylabel="Metric Value",
           title="TAGFC vs CoMTE vs AB-CF — CMAPSS FD001")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "F8_metric_comparison", c)


# ─────────────────────────────────────────────────────────────────────────────
# F9 — Scatter: proximity_l2 vs validity
# ─────────────────────────────────────────────────────────────────────────────

def plot_proximity_validity_scatter(results_dict: dict[str, list[dict]],
                                    c: Config = cfg) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, results in results_dict.items():
        l2s = [r["proximity_l2"] for r in results]
        vls = [r["validity"]     for r in results]
        ax.scatter(l2s, vls, label=name, alpha=0.5,
                   color=METHOD_COLORS.get(name, "gray"), s=40)

    ax.set(xlabel="L2 Distance to Original (lower = better)",
           ylabel="Validity (1 = flipped)",
           title="Proximity vs Validity Trade-off")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "F9_proximity_validity_scatter", c)


# ─────────────────────────────────────────────────────────────────────────────
# F10 — Omega (penalty weight) heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_omega_heatmap(omega: np.ndarray,
                       sensor_names: tuple = cfg.SENSOR_NAMES,
                       sample_idx: int = 0, c: Config = cfg) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(omega, aspect="auto", cmap="viridis_r", vmin=0, vmax=1)
    ax.set(yticks=range(len(sensor_names)),
           yticklabels=sensor_names,
           xlabel="Wavelet Coefficient Index",
           ylabel="Sensor",
           title=f"Omega (Penalty Weights) — Sample {sample_idx}\n"
                 f"Dark = Important (low cost), Light = Unimportant (high cost)")
    plt.colorbar(im, ax=ax, label="ω (0=important, 1=unimportant)")
    fig.tight_layout()
    _save(fig, f"F10_omega_heatmap_sample{sample_idx}", c)
