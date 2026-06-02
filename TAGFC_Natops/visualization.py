"""
visualization.py — Publication-quality figures for TAGFC on NATOPS.

Figures produced:
    F1   Training curves (loss + accuracy)
    F2   Confusion matrix (6×6)
    F3   Class distribution (train vs test)
    F4   Original vs CF gesture time series (all 24 channels)
    F5   Temporal saliency (attention rollout)
    F6   Channel importance bar chart
    F7   Frequency importance (Haar DWT bands)
    F8   Delta heatmap (channel × wavelet coefficient)
    F9   Omega heatmap (penalty weights)
    F10  Metric comparison bar chart (TAGFC vs CoMTE vs AB-CF)
    F11  Scatter: proximity_l2 vs validity (all methods)
    F12  Per-class validity breakdown
    F13  Optimisation convergence (loss terms per iteration)
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config, cfg

CLASS_NAMES = {
    0: "I have command",
    1: "All clear",
    2: "Not clear",
    3: "Spread wings",
    4: "Fold wings",
    5: "Lock wings",
}
CLASS_COLORS = {
    0: "#3498db", 1: "#2ecc71", 2: "#e74c3c",
    3: "#f39c12", 4: "#9b59b6", 5: "#1abc9c",
}
METHOD_COLORS = {"TAGFC": "#3498db", "CoMTE": "#e67e22", "AB-CF": "#9b59b6"}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     150,
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
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", color="#3498db")
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="#e74c3c")
    ax2.set(xlabel="Epoch", ylabel="Accuracy",
            title="Classification Accuracy", ylim=(0, 1.05))
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("TAGFC-NATOPS — TransformerNATOPS Training", fontweight="bold")
    fig.tight_layout()
    _save(fig, "F1_training_curves", c)


# ─────────────────────────────────────────────────────────────────────────────
# F2 — Confusion matrix (6×6)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          c: Config = cfg) -> None:
    K  = c.K
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    short = ["Cmd", "Clear", "NClear", "Spread", "Fold", "Lock"]
    ax.set(xticks=range(K), yticks=range(K),
           xticklabels=short, yticklabels=short,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix — TransformerNATOPS (6-class)")

    thresh = cm.max() / 2
    for i in range(K):
        for j in range(K):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11, fontweight="bold")

    fig.tight_layout()
    _save(fig, "F2_confusion_matrix", c)


# ─────────────────────────────────────────────────────────────────────────────
# F3 — Class distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(y_tr: np.ndarray, y_te: np.ndarray,
                             c: Config = cfg) -> None:
    short  = ["Cmd", "Clear", "NClear", "Spread", "Fold", "Lock"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, y, title in zip(axes, [y_tr, y_te], ["Train", "Test"]):
        counts = np.bincount(y, minlength=c.K)
        colors = [CLASS_COLORS[i] for i in range(c.K)]
        bars   = ax.bar(short, counts, color=colors, edgecolor="white")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(cnt), ha="center", va="bottom", fontsize=10)
        ax.set(title=f"{title} Class Distribution", ylabel="Count")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("NATOPS — Class Distribution", fontweight="bold")
    fig.tight_layout()
    _save(fig, "F3_class_distribution", c)


# ─────────────────────────────────────────────────────────────────────────────
# F4 — Original vs CF time series (all 24 channels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cf_timeseries(X_orig: np.ndarray, X_cf: np.ndarray,
                       orig_class: int, cf_class: int,
                       sensor_names: tuple = cfg.SENSOR_NAMES,
                       sample_idx: int = 0,
                       c: Config = cfg) -> None:
    T, D  = X_orig.shape
    cols  = 6
    rows  = (D + cols - 1) // cols
    t     = np.arange(T)

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.5), sharex=True)
    axes = axes.flatten()

    for d in range(D):
        ax = axes[d]
        ax.plot(t, X_orig[:, d], color="#2c3e50", lw=1.5, label="Original")
        ax.plot(t, X_cf[:, d],   color="#e74c3c",  lw=1.5, linestyle="--", label="CF")
        ax.fill_between(t, X_orig[:, d], X_cf[:, d], alpha=0.15, color="#e74c3c")
        ax.set_title(sensor_names[d], fontsize=8)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7)

    for ax in axes[D:]:
        ax.set_visible(False)

    orig_name = CLASS_NAMES.get(orig_class, str(orig_class))
    cf_name   = CLASS_NAMES.get(cf_class,   str(cf_class))
    fig.suptitle(
        f"TAGFC CF — Sample {sample_idx}  |  {orig_name}  →  {cf_name}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, f"F4_cf_timeseries_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F5 — Temporal saliency (attention rollout)
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
    ax.set(xlabel="Timestep", ylabel="Attention Weight",
           title=f"Temporal Saliency (Attention Rollout) — Sample {sample_idx}",
           ylim=(0, 1.05))
    ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label="Saliency")
    fig.tight_layout()
    _save(fig, f"F5_temporal_saliency_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F6 — Channel importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensor_importance(sensor_imp: np.ndarray,
                           sensor_names: tuple = cfg.SENSOR_NAMES,
                           sample_idx: int = 0,
                           c: Config = cfg) -> None:
    D   = len(sensor_imp)
    idx = np.argsort(sensor_imp)[::-1]

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["#e74c3c" if sensor_imp[i] > sensor_imp.mean() else "#95a5a6"
              for i in idx]
    ax.bar(range(D), sensor_imp[idx], color=colors)
    ax.set(xticks=range(D),
           xticklabels=[sensor_names[i] for i in idx],
           ylabel="Σ|Δ wavelet coeff|",
           title=f"Channel Importance (Wavelet Perturbation) — Sample {sample_idx}")
    ax.axhline(sensor_imp.mean(), color="#2c3e50", linestyle="--",
               alpha=0.6, label="Mean")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    _save(fig, f"F6_sensor_importance_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F7 — Frequency importance (Haar DWT bands)
# ─────────────────────────────────────────────────────────────────────────────

def plot_freq_importance(freq_imp: np.ndarray, wav_levels: int = 3,
                         sample_idx: int = 0, c: Config = cfg) -> None:
    L   = len(freq_imp)
    n_a = L >> wav_levels

    bands = [(0, n_a, "Approx (slow motion)", "#e74c3c")]
    size  = n_a
    for lvl in range(wav_levels):
        label = f"Detail-L{wav_levels - lvl}"
        col   = ["#f39c12", "#3498db", "#2ecc71"][lvl % 3]
        bands.append((n_a + lvl * size, n_a + (lvl + 1) * size, label, col))
        size *= 2

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(range(L), freq_imp, color="#bdc3c7", width=1.0, edgecolor="none")
    for start, end, label, col in bands:
        seg = freq_imp[start:end]
        ax.bar(range(start, start + len(seg)), seg, color=col,
               width=1.0, edgecolor="none", label=label, alpha=0.85)

    ax.set(xlabel="Wavelet Coefficient Index", ylabel="Σ|Δ coeff|",
           title=f"Frequency Importance (Haar DWT Bands) — Sample {sample_idx}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, f"F7_freq_importance_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F8 — Delta heatmap (channel × wavelet coefficient)
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_heatmap(delta: np.ndarray,
                       sensor_names: tuple = cfg.SENSOR_NAMES,
                       sample_idx: int = 0, c: Config = cfg) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    vmax = np.abs(delta).max() + 1e-8
    im   = ax.imshow(np.abs(delta), aspect="auto", cmap="hot", vmin=0, vmax=vmax)
    ax.set(yticks=range(len(sensor_names)),
           yticklabels=sensor_names,
           xlabel="Wavelet Coefficient Index",
           ylabel="Channel",
           title=f"|Δ| Heatmap (Channel × Frequency) — Sample {sample_idx}")
    plt.colorbar(im, ax=ax, label="|Δ coefficient|")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _save(fig, f"F8_delta_heatmap_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F9 — Omega heatmap (penalty weights)
# ─────────────────────────────────────────────────────────────────────────────

def plot_omega_heatmap(omega: np.ndarray,
                       sensor_names: tuple = cfg.SENSOR_NAMES,
                       sample_idx: int = 0, c: Config = cfg) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(omega, aspect="auto", cmap="viridis_r", vmin=0, vmax=1)
    ax.set(yticks=range(len(sensor_names)),
           yticklabels=sensor_names,
           xlabel="Wavelet Coefficient Index",
           ylabel="Channel",
           title=(f"Omega (Penalty Weights) — Sample {sample_idx}\n"
                  "Dark=Important (high cost), Light=Unimportant (low cost)"))
    plt.colorbar(im, ax=ax, label="ω (0=important, 1=unimportant)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _save(fig, f"F9_omega_heatmap_sample{sample_idx}", c)


# ─────────────────────────────────────────────────────────────────────────────
# F10 — Metric comparison bar chart (TAGFC vs CoMTE vs AB-CF)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_comparison(agg_tagfc: dict, agg_comte: dict, agg_abcf: dict,
                           c: Config = cfg) -> None:
    metrics = ["validity", "proximity_l1", "sparsity", "cf_confidence", "rcf"]
    labels  = ["Validity ↑", "Prox L1 ↓", "Sparsity ↑", "CF Conf ↑", "RCF ↓"]
    methods = {"TAGFC": agg_tagfc, "CoMTE": agg_comte, "AB-CF": agg_abcf}
    x, w    = np.arange(len(metrics)), 0.25

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
           title="TAGFC vs CoMTE vs AB-CF — NATOPS Benchmark")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "F10_metric_comparison", c)


# ─────────────────────────────────────────────────────────────────────────────
# F11 — Scatter: proximity_l2 vs validity
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
           ylabel="Validity (1 = class flipped)",
           title="Proximity vs Validity Trade-off — NATOPS")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "F11_proximity_validity_scatter", c)


# ─────────────────────────────────────────────────────────────────────────────
# F12 — Per-class validity breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_validity(results_dict: dict[str, list[dict]],
                            c: Config = cfg) -> None:
    K       = c.K
    short   = ["Cmd", "Clear", "NClear", "Spread", "Fold", "Lock"]
    methods = list(results_dict.keys())
    x       = np.arange(K)
    w       = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(methods):
        class_valid = {k: [] for k in range(K)}
        for r in results_dict[name]:
            class_valid[r.get("orig_class", 0)].append(r["validity"])
        rates = [np.mean(class_valid[k]) if class_valid[k] else 0.0 for k in range(K)]
        ax.bar(x + (i - 1) * w, rates, w,
               color=METHOD_COLORS.get(name, "gray"),
               label=name, alpha=0.85, edgecolor="white")

    ax.set(xticks=x, xticklabels=short,
           ylabel="Validity Rate", ylim=(0, 1.15),
           title="Per-Class Validity Rate — NATOPS")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "F12_per_class_validity", c)


# ─────────────────────────────────────────────────────────────────────────────
# F13 — Optimisation convergence (TAGFC only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(info: dict, sample_idx: int = 0, c: Config = cfg) -> None:
    history = info.get("history", [])
    if not history:
        return

    iters    = [h["iter"]       for h in history]
    l_flip   = [h["L_flip"]     for h in history]
    l_sparse = [h["L_sparse"]   for h in history]
    l_cross  = [h["L_cross"]    for h in history]
    l_mfld   = [h["L_manifold"] for h in history]
    total    = [h["total"]      for h in history]
    cf_conf  = [h["cf_conf"]    for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(iters, l_flip,   label="L_flip",     color="#e74c3c",  lw=1.5)
    axes[0].plot(iters, l_sparse, label="L_sparse",   color="#f39c12",  lw=1.5)
    axes[0].plot(iters, l_cross,  label="L_cross",    color="#3498db",  lw=1.5)
    axes[0].plot(iters, l_mfld,   label="L_manifold", color="#9b59b6",  lw=1.5)
    axes[0].plot(iters, total,    label="Total",       color="#2c3e50",  lw=2.0, linestyle="--")
    axes[0].set(ylabel="Loss", title=f"TAGFC Optimisation Convergence — Sample {sample_idx}")
    axes[0].legend(ncol=3); axes[0].grid(alpha=0.3)

    axes[1].plot(iters, cf_conf, color="#2ecc71", lw=2.0)
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="0.5 threshold")
    flipped_iters = [h["iter"] for h in history if h["flipped"]]
    if flipped_iters:
        axes[1].axvline(flipped_iters[0], color="red", linestyle="--",
                        alpha=0.7, label=f"First flip @ iter {flipped_iters[0]}")
    axes[1].set(xlabel="Iteration", ylabel="P(target class)",
                title="CF Confidence Trajectory", ylim=(0, 1.05))
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, f"F13_convergence_sample{sample_idx}", c)
