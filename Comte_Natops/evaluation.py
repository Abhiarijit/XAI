"""
evaluation.py — Counterfactual quality metrics for CoMTE on NATOPS.

Metrics computed per (X_orig, X_cf) pair:
    validity       : 1 if model(X_cf) == y_tgt, else 0
    proximity_l1   : mean absolute deviation  ||X_cf - X||_1 / (T*D)
    proximity_l2   : root mean squared deviation  ||X_cf - X||_2 / sqrt(T*D)
    proximity_linf : max absolute deviation  ||X_cf - X||_inf
    sparsity       : fraction of (timestep, channel) pairs unchanged
    compactness    : fraction of channels unchanged  (CoMTE-specific)
    plausibility   : L2 distance to nearest training sample of target class
    efficiency     : n_channels_swapped (lower = better for CoMTE)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def compute_metrics(X_orig:  np.ndarray,
                    X_cf:    np.ndarray,
                    model,
                    y_tgt:   int,
                    y_orig:  int,
                    X_train: np.ndarray,
                    y_train: np.ndarray) -> dict:
    """
    Compute all CF quality metrics.

    Parameters
    ----------
    X_orig  : (T, D) original time series
    X_cf    : (T, D) counterfactual time series
    model   : classifier with predict_proba(X[None]) → (1, K)
    y_tgt   : target class
    y_orig  : original class
    X_train : (N_tr, T, D) training data
    y_train : (N_tr,) training labels

    Returns
    -------
    dict of metric name → float value
    """
    T, D = X_orig.shape
    N    = T * D
    delta = X_cf - X_orig   # (T, D)

    # ── Validity ──────────────────────────────────────────────────────────────
    proba    = model.predict_proba(X_cf[None])[0]
    pred     = int(np.argmax(proba))
    validity = float(pred == y_tgt)

    # ── Proximity ─────────────────────────────────────────────────────────────
    prox_l1   = float(np.abs(delta).mean())
    prox_l2   = float(np.sqrt((delta ** 2).mean()))
    prox_linf = float(np.abs(delta).max())

    # ── Sparsity (element-level) ──────────────────────────────────────────────
    n_changed = float(np.sum(np.abs(delta) > 1e-8))
    sparsity  = 1.0 - (n_changed / N)

    # ── Compactness (channel-level — CoMTE primary metric) ────────────────────
    chan_changed  = float(np.sum(np.any(np.abs(delta) > 1e-8, axis=0)))
    compactness   = 1.0 - (chan_changed / D)

    # ── Plausibility (L2 to nearest target-class training sample) ────────────
    mask   = y_train == y_tgt
    X_tgt  = X_train[mask]
    if len(X_tgt) > 0:
        cf_flat   = X_cf.flatten()
        tgt_flat  = X_tgt.reshape(len(X_tgt), -1)
        dists     = np.linalg.norm(tgt_flat - cf_flat, axis=1)
        plausibility = float(dists.min())
    else:
        plausibility = float("nan")

    # ── Efficiency (channels swapped) ─────────────────────────────────────────
    efficiency = float(chan_changed)   # fewer = more efficient

    return {
        "validity"    : validity,
        "proximity_l1": prox_l1,
        "proximity_l2": prox_l2,
        "proximity_linf": prox_linf,
        "sparsity"    : sparsity,
        "compactness" : compactness,
        "plausibility": plausibility,
        "efficiency"  : efficiency,
        "pred_class"  : pred,
        "target_class": y_tgt,
        "orig_class"  : y_orig,
        "conf_target" : float(proba[y_tgt]),
    }


def aggregate(results: list[dict]) -> dict:
    """Compute mean ± std for each metric across all samples."""
    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    agg  = {}
    for k in keys:
        vals = [r[k] for r in results if not np.isnan(r[k])]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def print_summary(agg: dict, method: str = "CoMTE") -> None:
    print(f"\n{'─'*55}")
    print(f"  {method} — Aggregate Metrics  (mean ± std)")
    print(f"{'─'*55}")
    key_order = [
        "validity", "proximity_l1", "proximity_l2", "proximity_linf",
        "sparsity", "compactness", "plausibility", "efficiency",
    ]
    for k in key_order:
        if k in agg:
            v = agg[k]
            print(f"  {k:<20s}: {v['mean']:8.4f}  ±  {v['std']:.4f}")
    print(f"{'─'*55}")


def run_wilcoxon(results_a: list[dict],
                 results_b: list[dict],
                 label_b:   str = "Baseline",
                 alpha:     float = 0.05) -> dict:
    """
    Wilcoxon signed-rank test comparing two sets of per-sample metrics.

    For proximity and efficiency: lower is better (test if A < B).
    For validity, sparsity, compactness: higher is better (test if A > B).
    """
    metrics_lower_better = ["proximity_l1", "proximity_l2", "proximity_linf",
                             "plausibility", "efficiency"]
    metrics_higher_better = ["validity", "sparsity", "compactness"]

    print(f"\n  Wilcoxon: CoMTE vs {label_b}  (α={alpha})")
    print(f"  {'Metric':<20s}  {'stat':>8}  {'p-value':>10}  {'sig':>5}")
    print(f"  {'─'*50}")

    report = {}
    for k in metrics_lower_better + metrics_higher_better:
        a_vals = [r[k] for r in results_a if k in r and not np.isnan(r[k])]
        b_vals = [r[k] for r in results_b if k in r and not np.isnan(r[k])]
        n = min(len(a_vals), len(b_vals))
        if n < 5:
            continue
        try:
            stat, p = stats.wilcoxon(a_vals[:n], b_vals[:n])
            sig  = "**" if p < alpha else "  "
            report[k] = {"stat": float(stat), "p": float(p), "significant": p < alpha}
            print(f"  {k:<20s}  {stat:8.2f}  {p:10.4f}  {sig}")
        except Exception:
            pass

    return report
