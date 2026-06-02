"""
baselines.py — CoMTE and AB-CF counterfactual baselines.

CoMTE  — Ates et al., ICAPAI 2021
AB-CF  — Li et al., DaWaK 2023
"""
from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────────────────────

def _find_nun(X: np.ndarray, y_orig: int,
              X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Nearest Unlike Neighbour: closest training sample NOT of class y_orig."""
    mask    = y_train != y_orig
    X_other = X_train[mask]
    dists   = np.linalg.norm(
        (X_other - X[None]).reshape(len(X_other), -1), axis=1
    )
    return X_other[dists.argmin()]


# ─────────────────────────────────────────────────────────────────────────────
# CoMTE
# ─────────────────────────────────────────────────────────────────────────────

class CoMTE:
    """
    CoMTE: greedy NUN-based channel-swap counterfactual.

    At each step picks the single untouched sensor channel whose replacement
    (from the NUN) increases P(y_tgt | X_cf) the most, until a flip occurs
    or the budget is exhausted.
    """

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray,
                 max_channels: int | None = None):
        """
        model        : TransformerFD001
        X_train      : (N, T, D)
        y_train      : (N,)
        max_channels : maximum sensor channels to swap (None = all D)
        """
        self.model        = model
        self.X_train      = X_train
        self.y_train      = y_train
        self.max_channels = max_channels

    def explain(self, X: np.ndarray, y_tgt: int) -> tuple[np.ndarray, dict]:
        """
        X     : (T, D) normalised query window
        y_tgt : int  target class

        Returns
        -------
        X_cf  : (T, D) counterfactual
        info  : dict with metrics
        """
        T, D   = X.shape
        y_orig = self.model.predict(X)
        X_nun  = _find_nun(X, y_orig, self.X_train, self.y_train)

        budget  = self.max_channels if self.max_channels else D
        X_cf    = X.copy()
        swapped : set[int] = set()

        for _ in range(budget):
            best_conf = -1.0
            best_d    = -1

            for d in range(D):
                if d in swapped:
                    continue
                X_try = X_cf.copy()
                X_try[:, d] = X_nun[:, d]
                p = self.model.predict_proba(X_try)[y_tgt]
                if p > best_conf:
                    best_conf = p
                    best_d    = d

            if best_d < 0:
                break

            X_cf[:, best_d] = X_nun[:, best_d]
            swapped.add(best_d)

            if self.model.predict(X_cf) == y_tgt:
                break

        probs      = self.model.predict_proba(X_cf)
        final_pred = int(probs.argmax())

        return X_cf, {
            "flipped":         final_pred == y_tgt,
            "orig_class":      y_orig,
            "cf_class":        final_pred,
            "cf_confidence":   float(probs[y_tgt]),
            "channels_swapped": len(swapped),
        }


# ─────────────────────────────────────────────────────────────────────────────
# AB-CF
# ─────────────────────────────────────────────────────────────────────────────

class ABCF:
    """
    AB-CF: entropy-guided segment-swap counterfactual.

    Scores every (sensor, time-segment) pair by the Shannon entropy of the
    model's output when only that segment is unmasked.  Lowest-entropy pairs
    (most discriminative) are swapped from the NUN first.
    """

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray,
                 window_frac: float = 0.1):
        """
        model        : TransformerFD001
        X_train      : (N, T, D)
        y_train      : (N,)
        window_frac  : segment length as fraction of T (default 10 %)
        """
        self.model       = model
        self.X_train     = X_train
        self.y_train     = y_train
        self.window_frac = window_frac

    def explain(self, X: np.ndarray, y_tgt: int) -> tuple[np.ndarray, dict]:
        """
        X     : (T, D) normalised query window
        y_tgt : int  target class

        Returns
        -------
        X_cf  : (T, D) counterfactual
        info  : dict with metrics
        """
        T, D   = X.shape
        L      = max(1, int(self.window_frac * T))
        y_orig = self.model.predict(X)
        X_nun  = _find_nun(X, y_orig, self.X_train, self.y_train)

        # Score segments: zero-pad all other positions, compute entropy
        candidates: list[tuple[float, int, int]] = []
        for d in range(D):
            for start in range(0, T - L + 1, L):
                probe = np.zeros((T, D), dtype=np.float32)
                probe[start : start + L, d] = X[start : start + L, d]
                probs = self.model.predict_proba(probe)
                H     = float(-np.sum(probs * np.log2(probs + 1e-8)))
                candidates.append((H, d, start))

        candidates.sort()   # ascending entropy → most discriminative first

        X_cf = X.copy()
        for _, d, start in candidates:
            X_cf[start : start + L, d] = X_nun[start : start + L, d]
            if self.model.predict(X_cf) == y_tgt:
                break

        probs      = self.model.predict_proba(X_cf)
        final_pred = int(probs.argmax())

        return X_cf, {
            "flipped":       final_pred == y_tgt,
            "orig_class":    y_orig,
            "cf_class":      final_pred,
            "cf_confidence": float(probs[y_tgt]),
        }
