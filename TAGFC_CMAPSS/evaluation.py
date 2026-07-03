"""
evaluation.py — Metrics, aggregation, and Wilcoxon tests for TAGFC_CMAPSS.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

CLASS_NAMES = {0: "Healthy", 1: "Degrading", 2: "Critical"}

_HIGHER_BETTER = {"validity", "sparsity", "cf_confidence"}
_METRICS = [
    "validity", "proximity_l1", "proximity_l2", "proximity_linf",
    "sparsity", "coherence", "cf_confidence", "rcf",
]


def compute_metrics(X_orig, X_cf, model, y_tgt, y_orig,
                    Sigma_inv, X_train, y_train) -> dict:
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
    cf_conf   = float(probs[y_tgt])

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


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
    out  = {}
    for k in keys:
        vals = [r[k] for r in results if not np.isnan(r.get(k, float("nan")))]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    return out


def print_summary(agg: dict, method_name: str = "TAGFC") -> None:
    print(f"\n{'─'*55}\n  {method_name} Results\n{'─'*55}")
    for metric, v in agg.items():
        arrow = "↑" if metric in _HIGHER_BETTER else "↓"
        print(f"  {metric:18s} {v['mean']:8.4f} ± {v['std']:.4f}  {arrow}")
    print(f"{'─'*55}")


def run_wilcoxon(tagfc_results, baseline_results, baseline_name="CoMTE",
                 alpha=0.05) -> dict:
    n_tests    = 6   # Bonferroni over 6 continuous metrics
    alpha_corr = alpha / n_tests
    print(f"\n{'='*65}")
    print(f"  Wilcoxon: TAGFC vs {baseline_name}  (Bonferroni α={alpha_corr:.5f})")
    print(f"{'='*65}")

    cont_metrics = [m for m in _METRICS if m != "validity"]
    output = {}
    for metric in cont_metrics:
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
            stat, pval = wilcoxon(tv, bv, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            print(f"  {metric:<18s}  all differences zero, skipped")
            continue

        sig          = pval < alpha_corr
        tagfc_better = (tv.mean() > bv.mean()) if metric in _HIGHER_BETTER \
                       else (tv.mean() < bv.mean())
        marker = ("*** TAGFC" if (sig and tagfc_better)
                  else "(!!) baseline" if (sig and not tagfc_better) else "")
        print(f"  {metric:<18s}  TAGFC={tv.mean():.4f}  {baseline_name}={bv.mean():.4f}"
              f"  p={pval:.4f}  {marker}")
        output[metric] = {
            "stat": float(stat), "pval": float(pval),
            "significant": bool(sig), "tagfc_better": bool(tagfc_better),
            "tagfc_mean": float(tv.mean()), "tagfc_std": float(tv.std()),
            "baseline_mean": float(bv.mean()), "baseline_std": float(bv.std()),
        }
    print(f"{'='*65}\n")
    return output
