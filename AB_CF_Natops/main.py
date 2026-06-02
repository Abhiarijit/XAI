"""
main.py — AB-CF pipeline on NATOPS gesture dataset.

Steps
-----
1. Load & preprocess NATOPS data (TRAIN/TEST .arff)
2. Train (or load) TransformerNATOPS classifier
3. Run AB-CF on N_EXPLAIN test samples
4. Aggregate metrics + print summary
5. Generate all publication figures

Run:
    python main.py
"""
from __future__ import annotations

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import time
import numpy as np
import torch

from config import Config, cfg
from data_pipeline import load_natops, get_dataloaders
from transformer_model import (
    TransformerNATOPS, train_transformer, save_model, load_model, evaluate,
)
from abcf_core import ABCF
from evaluation import compute_metrics, aggregate, print_summary
from visualization import (
    plot_training_curves, plot_confusion_matrix, plot_class_distribution,
    plot_cf_timeseries, plot_entropy_heatmap, plot_channel_importance,
    plot_segment_importance, plot_swap_trace, plot_delta_heatmap,
    plot_nun_comparison, plot_metric_comparison,
    plot_proximity_validity_scatter, plot_per_class_validity,
)


def _banner(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def _target_class(y_orig: int, K: int = 6) -> int:
    """Cyclic target: class i → class (i+1) % K."""
    return (y_orig + 1) % K


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data
# ─────────────────────────────────────────────────────────────────────────────

def step1_data(c: Config = cfg):
    _banner("Step 1 — Load & Preprocess NATOPS Data")
    X_tr, y_tr, X_te, y_te, stats = load_natops(c)
    train_dl, test_dl = get_dataloaders(X_tr, y_tr, X_te, y_te, c)
    plot_class_distribution(y_tr, y_te, c)
    return X_tr, y_tr, X_te, y_te, stats, train_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Transformer
# ─────────────────────────────────────────────────────────────────────────────

def step2_transformer(train_dl, test_dl, c: Config = cfg):
    _banner("Step 2 — Train TransformerNATOPS Classifier")

    model_path = c.MODEL_DIR / "transformer_natops_abcf.pt"
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = TransformerNATOPS(c)

    if model_path.exists():
        print(f"  Found saved model at {model_path} — loading.")
        model  = load_model(model_path, c)
        crit   = torch.nn.CrossEntropyLoss()
        _, acc = evaluate(model, test_dl, crit, torch.device("cpu"))
        print(f"  Loaded val_acc = {acc:.4f}")
        history = None
    else:
        history = train_transformer(model, train_dl, test_dl, c, device)
        save_model(model, model_path)

    model.eval()
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — AB-CF Explanations
# ─────────────────────────────────────────────────────────────────────────────

def step3_abcf(model, X_tr, y_tr, X_te, y_te, c: Config = cfg):
    _banner("Step 3 — Generate Counterfactuals with AB-CF")

    rng = np.random.default_rng(c.RANDOM_SEED)

    # Balanced sample selection
    indices = []
    for cls in range(c.K):
        cls_idx = np.where(y_te == cls)[0]
        n_pick  = min(max(1, c.N_EXPLAIN // c.K), len(cls_idx))
        indices.extend(rng.choice(cls_idx, size=n_pick, replace=False).tolist())
    indices = indices[:c.N_EXPLAIN]

    print(f"  Explaining {len(indices)} test samples ...")
    print(f"  Config: n_segments={c.ABCF_N_SEGMENTS}, "
          f"n_bins={c.ABCF_N_BINS}, "
          f"window_frac={c.ABCF_WINDOW_FRAC}, "
          f"max_swaps={c.ABCF_MAX_SWAPS}")

    abcf = ABCF(
        model, X_tr, y_tr,
        n_segments  = c.ABCF_N_SEGMENTS,
        n_bins      = c.ABCF_N_BINS,
        window_frac = c.ABCF_WINDOW_FRAC,
        max_swaps   = c.ABCF_MAX_SWAPS,
    )

    results, cfs = [], []

    for n, idx in enumerate(indices):
        X      = X_te[idx]
        y_orig = int(y_te[idx])
        y_tgt  = _target_class(y_orig, c.K)

        print(f"\n  [{n+1:3d}/{len(indices)}] idx={idx}  "
              f"true={y_orig}→tgt={y_tgt}", end="  ", flush=True)

        t0         = time.time()
        X_cf, info = abcf.explain(X, y_tgt)
        elapsed    = time.time() - t0

        m = compute_metrics(X, X_cf, model, y_tgt, y_orig,
                            X_train=X_tr, y_train=y_tr)
        results.append(m)
        cfs.append((X, X_cf, info))

        print(f"AB-CF({elapsed:.1f}s, "
              f"valid={info['validity']}, "
              f"n_swaps={info['n_swaps']}, "
              f"P(tgt)={info['final_proba'][y_tgt]:.3f})")

    return results, cfs, abcf


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Metrics
# ─────────────────────────────────────────────────────────────────────────────

def step4_evaluate(results: list[dict], c: Config = cfg) -> dict:
    _banner("Step 4 — Aggregate Metrics")
    agg = aggregate(results)
    print_summary(agg, "AB-CF")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Figures
# ─────────────────────────────────────────────────────────────────────────────

def step5_figures(model, history,
                  X_tr, y_tr, X_te, y_te,
                  results, cfs, agg,
                  c: Config = cfg) -> None:
    _banner("Step 5 — Generate Publication Figures")

    if history is not None:
        plot_training_curves(history, c)

    y_pred = np.array([model.predict(X_te[i]) for i in range(len(y_te))])
    plot_confusion_matrix(y_te, y_pred, c)

    for sample_idx, (X_orig, X_cf, info) in enumerate(cfs[:3]):
        plot_cf_timeseries(X_orig, X_cf, info, c, sample_idx)
        plot_entropy_heatmap(info, c, sample_idx)
        plot_channel_importance(info, c, sample_idx)
        plot_segment_importance(info, c, sample_idx)
        plot_swap_trace(info, c, sample_idx)
        plot_delta_heatmap(X_orig, X_cf, info, c, sample_idx)
        plot_nun_comparison(X_orig, info["NUN"], X_cf, c, sample_idx)

    plot_metric_comparison(agg, c)
    plot_proximity_validity_scatter(results, c)
    plot_per_class_validity(results, c)

    print(f"\n  All figures saved to: {c.FIG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(c: Config = cfg):
    t_start = time.time()

    X_tr, y_tr, X_te, y_te, stats, train_dl, test_dl = step1_data(c)
    model, history = step2_transformer(train_dl, test_dl, c)
    results, cfs, abcf = step3_abcf(model, X_tr, y_tr, X_te, y_te, c)
    agg = step4_evaluate(results, c)
    step5_figures(model, history,
                  X_tr, y_tr, X_te, y_te,
                  results, cfs, agg, c)

    elapsed = time.time() - t_start
    _banner(f"AB-CF NATOPS pipeline complete  ({elapsed/60:.1f} min)")

    print(f"\n  Validity    : {agg['validity']['mean']*100:.1f}%")
    print(f"  Proximity L1: {agg['proximity_l1']['mean']:.4f}")
    print(f"  Sparsity    : {agg['sparsity']['mean']*100:.1f}%")
    print(f"  Compactness : {agg['compactness']['mean']*100:.1f}%")
    print(f"  Plausibility: {agg['plausibility']['mean']:.4f}")


if __name__ == "__main__":
    main()
