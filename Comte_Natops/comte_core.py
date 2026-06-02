"""
comte_core.py — CoMTE: Counterfactual Multivariate Time-series Explanations.

Reference: Ates et al. (2021) "Counterfactual Explanations for Multivariate
           Time Series", IEEE ICAC.

Algorithm:
-----------
Given query X (T, D) and target class y_tgt:

Step 1 — Find NUN (Nearest Unlike Neighbor):
    NUN = argmin_{x in X_train, y=y_tgt} DTW_distance(X, x)
    (Euclidean L2 used here for computational efficiency)

Step 2 — Greedy channel selection:
    For each swap iteration:
        For each unswapped channel c:
            Build candidate X_cand = X_cf with channel c replaced by NUN[:, c]
            Score = P(model predicts y_tgt | X_cand)
        Swap channel c* with highest score into X_cf

Step 3 — Repeat until:
    - Model predicts y_tgt (valid CF found), OR
    - All D channels have been swapped

Strategies implemented:
    "greedy"      — best-first: pick channel giving highest P(y_tgt) each step
    "sorted_desc" — sort channels by per-channel L2 distance (desc) then swap
    "sorted_asc"  — sort channels by per-channel L2 distance (asc) then swap
"""
from __future__ import annotations

import numpy as np
from config import Config, cfg


class CoMTE:
    """
    Counterfactual Multivariate Time-series Explanations (CoMTE).

    Parameters
    ----------
    model      : classifier with predict_proba(X: np.ndarray) → (N, K) and
                 predict(X: np.ndarray) → int
    X_train    : (N_tr, T, D) training samples
    y_train    : (N_tr,) training labels
    strategy   : "greedy" | "sorted_desc" | "sorted_asc"
    distance   : "euclidean" (channel-wise L2 distance for NUN search)
    """

    def __init__(self,
                 model,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 strategy: str = "greedy",
                 distance: str = "euclidean"):
        self.model    = model
        self.X_train  = X_train
        self.y_train  = y_train
        self.strategy = strategy
        self.distance = distance

    # ── NUN Search ────────────────────────────────────────────────────────────

    def _find_nun(self, X: np.ndarray, y_tgt: int) -> np.ndarray:
        """
        Find Nearest Unlike Neighbor: closest training sample of class y_tgt.

        X      : (T, D)
        Returns: (T, D)  nearest training sample with label y_tgt
        """
        mask = self.y_train == y_tgt
        X_tgt = self.X_train[mask]            # (M, T, D)

        # Flatten and compute L2 distances
        X_flat   = X.flatten()                # (T*D,)
        X_tgt_f  = X_tgt.reshape(len(X_tgt), -1)  # (M, T*D)
        dists    = np.linalg.norm(X_tgt_f - X_flat, axis=1)  # (M,)

        return X_tgt[np.argmin(dists)]        # (T, D)

    # ── Channel distance ──────────────────────────────────────────────────────

    def _channel_distances(self, X: np.ndarray, NUN: np.ndarray) -> np.ndarray:
        """
        Compute per-channel L2 distance between X and NUN.

        Returns: (D,) array of distances, one per channel.
        """
        return np.linalg.norm(X - NUN, axis=0)   # (D,)  norm over T axis

    # ── Core CoMTE ────────────────────────────────────────────────────────────

    def explain(self, X: np.ndarray, y_tgt: int) -> tuple[np.ndarray, dict]:
        """
        Generate counterfactual for X targeting class y_tgt.

        Parameters
        ----------
        X      : (T, D) query time series
        y_tgt  : target class index

        Returns
        -------
        X_cf   : (T, D) counterfactual time series
        info   : dict with explanation metadata
        """
        T, D = X.shape

        # Step 1: Find NUN
        NUN = self._find_nun(X, y_tgt)

        # Step 2: Determine swap order
        channel_dists = self._channel_distances(X, NUN)  # (D,)

        if self.strategy == "sorted_desc":
            swap_order = list(np.argsort(channel_dists)[::-1])  # most different first
        elif self.strategy == "sorted_asc":
            swap_order = list(np.argsort(channel_dists))         # most similar first
        else:
            swap_order = None   # greedy: determined dynamically

        # Step 3: Greedy channel swapping
        X_cf       = X.copy()
        swapped    = []
        prob_trace = []     # P(y_tgt) at each step

        remaining = list(range(D))

        for step in range(D):
            # Check if already valid
            proba = self.model.predict_proba(X_cf[None])[0]    # (K,)
            prob_trace.append(float(proba[y_tgt]))

            if np.argmax(proba) == y_tgt:
                break   # CF is valid — stop swapping

            if self.strategy == "greedy":
                # Find channel giving highest P(y_tgt) when swapped
                best_c     = None
                best_score = -1.0

                for c in remaining:
                    X_cand          = X_cf.copy()
                    X_cand[:, c]    = NUN[:, c]
                    p_tgt = float(self.model.predict_proba(X_cand[None])[0][y_tgt])
                    if p_tgt > best_score:
                        best_score = p_tgt
                        best_c     = c

                if best_c is None:
                    break
                c_swap = best_c

            else:
                # Pre-sorted order
                if not swap_order:
                    break
                c_swap = swap_order.pop(0)
                remaining.remove(c_swap)

            # Apply swap
            X_cf[:, c_swap] = NUN[:, c_swap]
            swapped.append(c_swap)

            if self.strategy == "greedy":
                remaining.remove(c_swap)

        # Final prediction
        final_proba = self.model.predict_proba(X_cf[None])[0]
        pred_class  = int(np.argmax(final_proba))
        valid       = (pred_class == y_tgt)

        # Sparsity: fraction of channels unchanged
        n_changed   = len(swapped)
        sparsity    = 1.0 - (n_changed / D)

        # Channel importance: 1 for swapped channels, 0 otherwise
        channel_importance = np.zeros(D)
        for c in swapped:
            channel_importance[c] = 1.0

        # Weighted importance by channel distance
        channel_importance_weighted = np.zeros(D)
        for c in swapped:
            channel_importance_weighted[c] = channel_dists[c]

        info = {
            "NUN"                        : NUN,
            "swapped_channels"           : swapped,
            "n_channels_swapped"         : n_changed,
            "sparsity"                   : sparsity,
            "prob_trace"                 : prob_trace,
            "final_proba"                : final_proba,
            "pred_class"                 : pred_class,
            "validity"                   : valid,
            "channel_distances"          : channel_dists,
            "channel_importance"         : channel_importance,
            "channel_importance_weighted": channel_importance_weighted,
            "orig_class"                 : int(self.model.predict(X)),
            "cf_class"                   : pred_class,
            "strategy"                   : self.strategy,
        }

        return X_cf, info


# ─────────────────────────────────────────────────────────────────────────────
# Batch explanation wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CoMTEBatch:
    """
    Wrapper to run CoMTE over a batch of test samples.

    Parameters
    ----------
    model      : classifier
    X_train    : (N_tr, T, D)
    y_train    : (N_tr,)
    c          : Config
    """

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray, c: Config = cfg):
        self.comte = CoMTE(
            model,
            X_train,
            y_train,
            strategy=c.COMTE_STRATEGY,
            distance=c.COMTE_DISTANCE,
        )
        self.c = c

    def explain_batch(self,
                      X_test:  np.ndarray,
                      y_test:  np.ndarray,
                      indices: list[int],
                      target_fn) -> tuple[list, list]:
        """
        Run CoMTE on selected test indices.

        Parameters
        ----------
        X_test    : (N_te, T, D)
        y_test    : (N_te,)
        indices   : list of test indices to explain
        target_fn : callable(y_orig, K) → y_tgt

        Returns
        -------
        results : list of metric dicts (one per sample)
        cfs     : list of (X_orig, X_cf, info) tuples
        """
        from evaluation import compute_metrics

        results, cfs = [], []

        for n, idx in enumerate(indices):
            X      = X_test[idx]
            y_orig = int(y_test[idx])
            y_tgt  = target_fn(y_orig, self.c.K)

            print(f"  [{n+1:3d}/{len(indices)}] idx={idx}  "
                  f"true={y_orig}→tgt={y_tgt}", end="  ", flush=True)

            import time
            t0 = time.time()
            X_cf, info = self.comte.explain(X, y_tgt)
            elapsed = time.time() - t0

            m = compute_metrics(X, X_cf, self.comte.model, y_tgt, y_orig,
                                X_train=self.comte.X_train, y_train=self.comte.y_train)
            results.append(m)
            cfs.append((X, X_cf, info))

            print(f"CoMTE({elapsed:.1f}s, "
                  f"valid={info['validity']}, "
                  f"swapped={info['n_channels_swapped']}/{self.c.D})")

        return results, cfs
