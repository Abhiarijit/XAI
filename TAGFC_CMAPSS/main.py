"""
main.py — Full TAGFC pipeline on NASA C-MAPSS FD001.

Steps
-----
1. Load & preprocess CMAPSS FD001
2. Train (or load) TransformerCMAPSS
3-5. Run TAGFC / CoMTE / AB-CF on N_EXPLAIN test engines
6. Aggregate metrics + Wilcoxon tests
7. Publication figures
8. Save results to outputs/results_cmapss.pkl
"""
from __future__ import annotations

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import pickle
import time
import numpy as np
from pathlib import Path

from config import Config, cfg
from data_pipeline import load_cmapss, get_dataloaders
from transformer_model import (TransformerCMAPSS, train_transformer,
                                save_model, load_model, evaluate)
from tagfc_core import TAGFCOptimizer
from baselines import CoMTE, ABCF
from evaluation import compute_metrics, aggregate, print_summary, run_wilcoxon
from visualization import (
    plot_training_curves, plot_confusion_matrix,
    plot_cf_timeseries, plot_temporal_saliency,
    plot_metric_comparison, plot_proximity_validity_scatter,
)

import torch


def _banner(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def _target_class(y_orig: int) -> int:
    """Critical→Healthy, Healthy→Critical, Degrading→Healthy."""
    if y_orig == 2:
        return 0   # Critical → Healthy
    elif y_orig == 0:
        return 2   # Healthy  → Critical (less common but tests the flip)
    else:
        return 0   # Degrading → Healthy


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data
# ─────────────────────────────────────────────────────────────────────────────

def step1_data(c: Config = cfg):
    _banner("Step 1 — Load & Preprocess C-MAPSS FD001")
    X_tr, y_tr, X_te, y_te, stats = load_cmapss(c)
    train_dl, test_dl = get_dataloaders(X_tr, y_tr, X_te, y_te, c)
    return X_tr, y_tr, X_te, y_te, stats, train_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Transformer
# ─────────────────────────────────────────────────────────────────────────────

def step2_transformer(train_dl, test_dl, c: Config = cfg):
    _banner("Step 2 — Train TransformerCMAPSS")
    model_path = c.MODEL_DIR / "transformer_cmapss.pt"
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = TransformerCMAPSS(c)
    if model_path.exists():
        print(f"  Found saved model at {model_path} — loading.")
        model   = load_model(model_path, c)
        crit    = torch.nn.CrossEntropyLoss()
        _, acc  = evaluate(model, test_dl, crit, torch.device("cpu"))
        print(f"  Loaded test acc = {acc:.4f}")
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

    # Balanced selection: prefer Critical and Degrading (more informative)
    indices = []
    n_per_class = c.N_EXPLAIN // c.K
    for cls in range(c.K):
        cls_idx = np.where(y_te == cls)[0]
        n_pick  = min(n_per_class, len(cls_idx))
        indices.extend(rng.choice(cls_idx, size=n_pick, replace=False).tolist())
    # top-up if needed
    remaining = c.N_EXPLAIN - len(indices)
    if remaining > 0:
        all_idx  = np.arange(len(y_te))
        used     = set(indices)
        leftover = [i for i in all_idx if i not in used]
        extra    = rng.choice(leftover, size=min(remaining, len(leftover)), replace=False)
        indices.extend(extra.tolist())
    indices = indices[:c.N_EXPLAIN]
    print(f"  Explaining {len(indices)} test engines …")

    tagfc = TAGFCOptimizer(model, X_tr, y_tr, c)
    comte = CoMTE(model, X_tr, y_tr)
    abcf  = ABCF(model, X_tr, y_tr)

    tagfc_results, comte_results, abcf_results = [], [], []
    tagfc_cfs = []

    for n, idx in enumerate(indices):
        X      = X_te[idx]
        y_orig = int(y_te[idx])
        y_tgt  = _target_class(y_orig)

        print(f"\n  [{n+1:3d}/{len(indices)}] engine={idx}  "
              f"class={y_orig}({cfg.CLASS_NAMES[y_orig]})→tgt={y_tgt}",
              end="  ", flush=True)

        t0 = time.time()
        X_cf_t, delta, t_info = tagfc.optimize(X, y_tgt)
        tagfc_cfs.append((X, X_cf_t, t_info))
        m_t = compute_metrics(X, X_cf_t, model, y_tgt, y_orig,
                              tagfc.Sigma_inv, X_tr, y_tr)
        tagfc_results.append(m_t)
        print(f"TAGFC({time.time()-t0:.1f}s,flip={m_t['validity']})", end="  ")

        t0 = time.time()
        X_cf_c, c_info = comte.explain(X, y_tgt)
        m_c = compute_metrics(X, X_cf_c, model, y_tgt, y_orig,
                              tagfc.Sigma_inv, X_tr, y_tr)
        comte_results.append(m_c)
        print(f"CoMTE({time.time()-t0:.1f}s,flip={m_c['validity']})", end="  ")

        t0 = time.time()
        X_cf_a, a_info = abcf.explain(X, y_tgt)
        m_a = compute_metrics(X, X_cf_a, model, y_tgt, y_orig,
                              tagfc.Sigma_inv, X_tr, y_tr)
        abcf_results.append(m_a)
        print(f"AB-CF({time.time()-t0:.1f}s,flip={m_a['validity']})")

    return tagfc_results, comte_results, abcf_results, tagfc_cfs, tagfc.Sigma_inv


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

def step7_figures(model, history, X_te, y_te,
                  tagfc_cfs, agg_t, agg_c, agg_a,
                  tagfc_results, comte_results, abcf_results,
                  c: Config = cfg):
    _banner("Step 7 — Publication Figures")

    if history is not None:
        plot_training_curves(history, c)

    y_pred = np.array([model.predict(X_te[i]) for i in range(len(y_te))])
    plot_confusion_matrix(y_te, y_pred, c)

    for sample_idx, (X_orig, X_cf, info) in enumerate(tagfc_cfs[:3]):
        plot_cf_timeseries(X_orig, X_cf,
                           info["orig_class"], info["cf_class"],
                           c.SENSOR_NAMES, sample_idx, c)
        plot_temporal_saliency(info["temporal_saliency"], sample_idx, c)

    plot_metric_comparison(agg_t, agg_c, agg_a, c)
    plot_proximity_validity_scatter(
        {"TAGFC": tagfc_results, "CoMTE": comte_results, "AB-CF": abcf_results},
        c,
    )
    print(f"\n  All figures saved to: {c.FIG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(c: Config = cfg):
    t_start = time.time()

    X_tr, y_tr, X_te, y_te, stats, train_dl, test_dl = step1_data(c)
    model, history = step2_transformer(train_dl, test_dl, c)

    (tagfc_results, comte_results, abcf_results,
     tagfc_cfs, Sigma_inv) = step345_explain(model, X_tr, y_tr, X_te, y_te, c)

    results_path = c.OUTPUT_DIR / "results_cmapss.pkl"
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
        model, history, X_te, y_te,
        tagfc_cfs, agg_t, agg_c, agg_a,
        tagfc_results, comte_results, abcf_results,
        c,
    )

    elapsed = time.time() - t_start
    _banner(f"Pipeline complete  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
