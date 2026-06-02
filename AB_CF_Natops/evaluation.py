"""
evaluation.py — Counterfactual quality metrics for AB-CF on NATOPS.

Metrics per (X_orig, X_cf) pair:
    validity       : 1 if model(X_cf) == y_tgt
    proximity_l1   : mean |X_cf - X| / (T*D)
    proximity_l2   : RMS deviation / sqrt(T*D)
    proximity_linf : max |X_cf - X|
    sparsity       : fraction of (timestep, channel) elements unchanged
    compactness    : fraction of channels with no change
    plausibility   : L2 distance to nearest target-class training sample
    efficiency     : number of (channel, segment) pairs swapped
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
    Parameters
    ----------
    X_orig  : (T, D)
    X_cf    : (T, D)
    model   : classifier with predict_proba
    y_tgt   : target class
    y_orig  : original class
    X_train : (N_tr, T, D)
    y_train : (N_tr,)

    Returns
    -------
    dict of metric name → float
    """
    T, D  = X_orig.shape
    N     = T * D
    delta = X_cf - X_orig

    # Validity
    proba    = model.predict_proba(X_cf[None])[0]
    pred     = int(np.argmax(proba))
    validity = float(pred == y_tgt)

    # Proximity
    prox_l1   = float(np.abs(delta).mean())
    prox_l2   = float(np.sqrt((delta ** 2).mean()))
    prox_linf = float(np.abs(delta).max())

    # Sparsity (element-level)
    n_elem_changed = float(np.sum(np.abs(delta) > 1e-8))
    sparsity       = 1.0 - (n_elem_changed / N)

    # Compactness (channel-level)
    n_chan_changed = float(np.sum(np.any(np.abs(delta) > 1e-8, axis=0)))
    compactness    = 1.0 - (n_chan_changed / D)

    # Plausibility
    mask  = y_train == y_tgt
    X_tgt = X_train[mask]
    if len(X_tgt) > 0:
        cf_flat  = X_cf.flatten()
        tgt_flat = X_tgt.reshape(len(X_tgt), -1)
        dists    = np.linalg.norm(tgt_flat - cf_flat, axis=1)
        plausibility = float(dists.min())
    else:
        plausibility = float("nan")

    # Efficiency: number of changed segments (approximate via changed channels)
    efficiency = float(n_chan_changed)

    return {
        "validity"      : validity,
        "proximity_l1"  : prox_l1,
        "proximity_l2"  : prox_l2,
        "proximity_linf": prox_linf,
        "sparsity"      : sparsity,
        "compactness"   : compactness,
        "plausibility"  : plausibility,
        "efficiency"    : efficiency,
        "pred_class"    : pred,
        "target_class"  : y_tgt,
        "orig_class"    : y_orig,
        "conf_target"   : float(proba[y_tgt]),
    }


def aggregate(results: list[dict]) -> dict:
    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    agg  = {}
    for k in keys:
        vals = [r[k] for r in results if not np.isnan(r[k])]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def print_summary(agg: dict, method: str = "AB-CF") -> None:
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
    metrics_lower  = ["proximity_l1", "proximity_l2", "proximity_linf",
                      "plausibility", "efficiency"]
    metrics_higher = ["validity", "sparsity", "compactness"]

    print(f"\n  Wilcoxon: AB-CF vs {label_b}  (α={alpha})")
    print(f"  {'Metric':<20s}  {'stat':>8}  {'p-value':>10}  {'sig':>5}")
    print(f"  {'─'*50}")

    report = {}
    for k in metrics_lower + metrics_higher:
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
