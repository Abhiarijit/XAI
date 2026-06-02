"""
main.py — TAGFC pipeline on NATOPS gesture dataset.

Steps
-----
1. Load & preprocess NATOPS data (TRAIN/TEST .arff)
2. Train (or load) TransformerNATOPS
3. Run TAGFC on N_EXPLAIN test samples
4. Run CoMTE on the same samples
5. Run AB-CF  on the same samples
6. Aggregate metrics + print summary tables
7. Wilcoxon significance tests (TAGFC vs CoMTE, TAGFC vs AB-CF)
8. Generate all publication figures

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
import pickle
import numpy as np
import torch

from config import Config, cfg
from data_pipeline import load_natops, get_dataloaders
from transformer_model import (
    TransformerNATOPS, train_transformer, save_model, load_model, evaluate,
)
from tagfc_core import TAGFCOptimizer
from baselines import CoMTE, ABCF
from evaluation import compute_metrics, aggregate, print_summary, run_wilcoxon
from visualization import (
    plot_training_curves, plot_confusion_matrix, plot_class_distribution,
    plot_cf_timeseries, plot_temporal_saliency,
    plot_sensor_importance, plot_freq_importance,
    plot_delta_heatmap, plot_omega_heatmap,
    plot_metric_comparison, plot_proximity_validity_scatter,
    plot_per_class_validity, plot_convergence,
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

    model_path = c.MODEL_DIR / "transformer_natops_tagfc.pt"
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
# Steps 3-5 — Explanation loop
# ─────────────────────────────────────────────────────────────────────────────

def step345_explain(model, X_tr, y_tr, X_te, y_te, c: Config = cfg):
    _banner("Steps 3-5 — Generate Counterfactuals (TAGFC / CoMTE / AB-CF)")

    rng = np.random.default_rng(c.RANDOM_SEED)

    # Balanced sample selection across 6 classes
    indices = []
    for cls in range(c.K):
        cls_idx = np.where(y_te == cls)[0]
        n_pick  = min(max(1, c.N_EXPLAIN // c.K), len(cls_idx))
        indices.extend(rng.choice(cls_idx, size=n_pick, replace=False).tolist())
    indices = indices[:c.N_EXPLAIN]
    print(f"  Explaining {len(indices)} test samples ...")

    tagfc = TAGFCOptimizer(model, X_tr, y_tr, c)
    comte = CoMTE(model, X_tr, y_tr)
    abcf  = ABCF(model,  X_tr, y_tr, window_frac=c.ABCF_WINDOW_FRAC)

    tagfc_results, comte_results, abcf_results = [], [], []
    tagfc_cfs,     comte_cfs,     abcf_cfs     = [], [], []

    for n, idx in enumerate(indices):
        X      = X_te[idx]
        y_orig = int(y_te[idx])
        y_tgt  = _target_class(y_orig, c.K)

        print(f"\n  [{n+1:3d}/{len(indices)}] idx={idx}  "
              f"true={y_orig}→tgt={y_tgt}", end="  ", flush=True)

        # ── TAGFC ─────────────────────────────────────────────────────────────
        t0 = time.time()
        X_cf_t, delta, t_info = tagfc.optimize(X, y_tgt)
        tagfc_cfs.append((X, X_cf_t, t_info))
        m_t = compute_metrics(X, X_cf_t, model, y_tgt, y_orig,
                              tagfc.Sigma_inv, X_tr, y_tr)
        tagfc_results.append(m_t)
        print(f"TAGFC({time.time()-t0:.1f}s, valid={t_info['validity']})", end="  ")

        # ── CoMTE ─────────────────────────────────────────────────────────────
        t0 = time.time()
        X_cf_c, c_info = comte.explain(X, y_tgt)
        comte_cfs.append((X, X_cf_c, c_info))
        m_c = compute_metrics(X, X_cf_c, model, y_tgt, y_orig,
                              tagfc.Sigma_inv, X_tr, y_tr)
        comte_results.append(m_c)
        print(f"CoMTE({time.time()-t0:.1f}s, valid={c_info['flipped']})", end="  ")

        # ── AB-CF ─────────────────────────────────────────────────────────────
        t0 = time.time()
        X_cf_a, a_info = abcf.explain(X, y_tgt)
        abcf_cfs.append((X, X_cf_a, a_info))
        m_a = compute_metrics(X, X_cf_a, model, y_tgt, y_orig,
                              tagfc.Sigma_inv, X_tr, y_tr)
        abcf_results.append(m_a)
        print(f"AB-CF({time.time()-t0:.1f}s, valid={a_info['flipped']})")

    return (tagfc_results, comte_results, abcf_results,
            tagfc_cfs,     comte_cfs,     abcf_cfs,
            tagfc.Sigma_inv)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Metrics & Wilcoxon
# ─────────────────────────────────────────────────────────────────────────────

def step6_evaluate(tagfc_results, comte_results, abcf_results, c: Config = cfg):
    _banner("Step 6 — Aggregate Metrics & Significance Tests")

    agg_t = aggregate(tagfc_results)
    agg_c = aggregate(comte_results)
    agg_a = aggregate(abcf_results)

    print_summary(agg_t, "TAGFC")
    print_summary(agg_c, "CoMTE")
    print_summary(agg_a, "AB-CF")

    wc_tc = run_wilcoxon(tagfc_results, comte_results, "CoMTE", c.ALPHA)
    wc_ta = run_wilcoxon(tagfc_results, abcf_results,  "AB-CF", c.ALPHA)

    return agg_t, agg_c, agg_a, wc_tc, wc_ta


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Figures
# ─────────────────────────────────────────────────────────────────────────────

def step7_figures(model, history,
                  X_tr, y_tr, X_te, y_te,
                  tagfc_cfs, agg_t, agg_c, agg_a,
                  tagfc_results, comte_results, abcf_results,
                  c: Config = cfg) -> None:
    _banner("Step 7 — Publication Figures")

    # F1 — Training curves
    if history is not None:
        plot_training_curves(history, c)

    # F2 — Confusion matrix
    y_pred = np.array([model.predict(X_te[i]) for i in range(len(y_te))])
    plot_confusion_matrix(y_te, y_pred, c)

    # F4-F9 + F13 — Per-sample TAGFC explanations (first 3 samples)
    for sample_idx, (X_orig, X_cf, info) in enumerate(tagfc_cfs[:3]):
        plot_cf_timeseries(X_orig, X_cf,
                           info["orig_class"], info["cf_class"],
                           c.SENSOR_NAMES, sample_idx, c)
        plot_temporal_saliency(info["temporal_saliency"], sample_idx, c)
        plot_sensor_importance(info["sensor_importance"], c.SENSOR_NAMES, sample_idx, c)
        plot_freq_importance(info["freq_importance"], c.WAV_LEVELS, sample_idx, c)
        plot_delta_heatmap(info["delta"], c.SENSOR_NAMES, sample_idx, c)
        plot_omega_heatmap(info["omega"], c.SENSOR_NAMES, sample_idx, c)
        plot_convergence(info, sample_idx, c)

    # F10 — Aggregate metric comparison
    plot_metric_comparison(agg_t, agg_c, agg_a, c)

    # F11 — Proximity-Validity scatter
    plot_proximity_validity_scatter(
        {"TAGFC": tagfc_results, "CoMTE": comte_results, "AB-CF": abcf_results},
        c,
    )

    # F12 — Per-class validity
    plot_per_class_validity(
        {"TAGFC": tagfc_results, "CoMTE": comte_results, "AB-CF": abcf_results},
        c,
    )

    print(f"\n  All figures saved to: {c.FIG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(c: Config = cfg):
    t_start = time.time()

    X_tr, y_tr, X_te, y_te, stats, train_dl, test_dl = step1_data(c)
    model, history = step2_transformer(train_dl, test_dl, c)

    (tagfc_results, comte_results, abcf_results,
     tagfc_cfs, comte_cfs, abcf_cfs,
     Sigma_inv) = step345_explain(model, X_tr, y_tr, X_te, y_te, c)

    # Save raw results
    results_path = c.OUTPUT_DIR / "results_natops_tagfc.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({
            "tagfc":     tagfc_results,
            "comte":     comte_results,
            "abcf":      abcf_results,
            "tagfc_cfs": tagfc_cfs,
        }, f)
    print(f"\n  Results saved → {results_path}")

    agg_t, agg_c, agg_a, wc_tc, wc_ta = step6_evaluate(
        tagfc_results, comte_results, abcf_results, c
    )

    step7_figures(
        model, history,
        X_tr, y_tr, X_te, y_te,
        tagfc_cfs, agg_t, agg_c, agg_a,
        tagfc_results, comte_results, abcf_results,
        c,
    )

    elapsed = time.time() - t_start
    _banner(f"TAGFC-NATOPS pipeline complete  ({elapsed/60:.1f} min)")

    # Final summary
    print(f"\n  TAGFC  Validity   : {agg_t['validity']['mean']*100:.1f}%")
    print(f"  CoMTE  Validity   : {agg_c['validity']['mean']*100:.1f}%")
    print(f"  AB-CF  Validity   : {agg_a['validity']['mean']*100:.1f}%")
    print(f"\n  TAGFC  Proximity L1: {agg_t['proximity_l1']['mean']:.4f}")
    print(f"  CoMTE  Proximity L1: {agg_c['proximity_l1']['mean']:.4f}")
    print(f"  AB-CF  Proximity L1: {agg_a['proximity_l1']['mean']:.4f}")
    print(f"\n  TAGFC  Sparsity   : {agg_t['sparsity']['mean']*100:.1f}%")
    print(f"  CoMTE  Sparsity   : {agg_c['sparsity']['mean']*100:.1f}%")
    print(f"  AB-CF  Sparsity   : {agg_a['sparsity']['mean']*100:.1f}%")


if __name__ == "__main__":
    main()
