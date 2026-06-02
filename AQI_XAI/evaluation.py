"""
evaluation.py — Metrics, aggregation, and Wilcoxon significance tests for AQI_XAI.

Metrics (per X_orig / X_cf pair):
    validity         — did CF flip the prediction?                (higher = better)
    proximity_l1     — L1 distance to original                    (lower  = better)
    proximity_l2     — L2 distance to original                    (lower  = better)
    proximity_linf   — L∞ distance to original                    (lower  = better)
    sparsity         — fraction of unchanged features             (higher = better)
    coherence        — Mahalanobis distance (inter-pollutant)     (lower  = better)
    cf_confidence    — model P(y_tgt | X_cf)                      (higher = better)
    rcf              — relative CF distance d(X,X*)/d(X,X_NUN)   (lower  = better)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

CLASS_NAMES = {0: "Good", 1: "Moderate", 2: "Poor"}


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(X_orig: np.ndarray,
                    X_cf:   np.ndarray,
                    model,
                    y_tgt:  int,
                    y_orig: int,
                    Sigma_inv: np.ndarray,
                    X_train: np.ndarray,
                    y_train: np.ndarray) -> dict:
    T, D = X_orig.shape
    diff = X_cf - X_orig

    probs      = model.predict_proba(X_cf)
    final_pred = int(probs.argmax())
    validity   = int(final_pred == y_tgt)

    l1   = float(np.abs(diff).sum())
    l2   = float(np.linalg.norm(diff))
    linf = float(np.abs(diff).max())

    sparsity  = float((np.abs(diff) < 1e-6).mean())

    mean_diff = diff.mean(axis=0)
    coherence = float(mean_diff @ Sigma_inv @ mean_diff)

    cf_conf = float(probs[y_tgt])

    mask    = y_train != y_orig
    X_other = X_train[mask]
    if len(X_other) > 0:
        d_nun = np.linalg.norm(
            (X_other - X_orig[None]).reshape(len(X_other), -1), axis=1
        ).min()
        rcf = l2 / (d_nun + 1e-8)
    else:
        rcf = float("nan")

    return {
        "validity":       validity,
        "proximity_l1":   l1,
        "proximity_l2":   l2,
        "proximity_linf": linf,
        "sparsity":       sparsity,
        "coherence":      coherence,
        "cf_confidence":  cf_conf,
        "rcf":            rcf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict[str, dict]:
    if not results:
        return {}
    keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
    out  = {}
    for k in keys:
        vals = [r[k] for r in results if not np.isnan(r.get(k, float("nan")))]
        out[k] = {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "values": vals,
        }
    return out


def print_summary(agg: dict, method_name: str = "TAGFC") -> None:
    higher_better = {"validity", "sparsity", "cf_confidence"}
    print(f"\n{'─'*55}")
    print(f"  {method_name} Results")
    print(f"{'─'*55}")
    for metric, v in agg.items():
        arrow = "↑" if metric in higher_better else "↓"
        print(f"  {metric:18s} {v['mean']:8.4f} ± {v['std']:.4f}  {arrow}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon signed-rank tests (Bonferroni corrected)
# ─────────────────────────────────────────────────────────────────────────────

_HIGHER_BETTER = {"validity", "sparsity", "cf_confidence"}
_METRICS = [
    "validity", "proximity_l1", "sparsity",
    "coherence", "cf_confidence", "rcf",
]


def run_wilcoxon(tagfc_results:    list[dict],
                 baseline_results: list[dict],
                 baseline_name:    str   = "CoMTE",
                 alpha:            float = 0.05) -> dict:
    """
    Paired Wilcoxon signed-rank tests (TAGFC vs baseline).
    Bonferroni correction: alpha / n_metrics.
    """
    n_tests    = len(_METRICS)
    alpha_corr = alpha / n_tests

    print(f"\n{'='*65}")
    print(f"  Wilcoxon Signed-Rank Tests:  TAGFC  vs  {baseline_name}")
    print(f"  Bonferroni α = {alpha:.3f}/{n_tests} = {alpha_corr:.5f}")
    print(f"{'='*65}")
    fmt = "  {:<18s}  TAGFC={:.4f}±{:.4f}  {:s}={:.4f}±{:.4f}  p={:.4f} {:s}"

    output = {}
    for metric in _METRICS:
        tv = np.array([r[metric] for r in tagfc_results
                       if not np.isnan(r.get(metric, float("nan")))])
        bv = np.array([r[metric] for r in baseline_results
                       if not np.isnan(r.get(metric, float("nan")))])

        n = min(len(tv), len(bv))
        if n < 5:
            print(f"  {metric:<18s}  skipped (n={n} < 5)")
            continue

        tv, bv = tv[:n], bv[:n]

        try:
            stat, pval = wilcoxon(tv, bv, zero_method="wilcox",
                                  alternative="two-sided")
        except ValueError:
            print(f"  {metric:<18s}  all differences zero, skipped")
            continue

        significant  = pval < alpha_corr
        tagfc_better = (tv.mean() > bv.mean()) if metric in _HIGHER_BETTER \
                       else (tv.mean() < bv.mean())

        if significant and tagfc_better:
            marker = "*** TAGFC better"
        elif significant and not tagfc_better:
            marker = "(!!) baseline better"
        else:
            marker = ""

        print(fmt.format(
            metric,
            tv.mean(), tv.std(),
            baseline_name, bv.mean(), bv.std(),
            pval, marker,
        ))

        output[metric] = {
            "stat":          float(stat),
            "pval":          float(pval),
            "significant":   bool(significant),
            "tagfc_better":  bool(tagfc_better),
            "tagfc_mean":    float(tv.mean()),
            "tagfc_std":     float(tv.std()),
            "baseline_mean": float(bv.mean()),
            "baseline_std":  float(bv.std()),
        }

    print(f"{'='*65}\n")
    return output
