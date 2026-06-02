"""
abcf_core.py — AB-CF: Attention-Based Counterfactual for Time Series.

Reference: Li et al. (2023) "AB-CF: Attention-Based Counterfactual Explanation
           for Multivariate Time Series Classification."

Algorithm (faithful reimplementation):
----------------------------------------
Given query X (T, D) and target class y_tgt:

Step 1 — Find NUN (Nearest Unlike Neighbor):
    NUN = argmin_{x in X_train, y=y_tgt} ||X - x||_2
    The NUN provides the "donor" segments for counterfactual construction.

Step 2 — Segment the time series:
    Divide T timesteps into n_seg equal-length segments.
    Segment k spans timesteps [k*seg_len : (k+1)*seg_len].

Step 3 — Compute Shannon entropy per (channel, segment) pair:
    For segment s of channel d in X:
        1. Discretise values into n_bins histogram bins
        2. Compute normalised probability distribution p_i
        3. H(s, d) = -sum(p_i * log2(p_i + eps))
    High entropy = high variability = more discriminative region.

Step 4 — Rank all (channel, segment) pairs by entropy (descending):
    Higher entropy segments are swapped first — they carry more information
    and are more likely to flip the class when replaced from NUN.

Step 5 — Incremental greedy swap:
    For each (channel c, segment s) in ranked order:
        Replace X_cf[seg_start:seg_end, c] = NUN[seg_start:seg_end, c]
        If model(X_cf) == y_tgt: stop (valid CF found)

    Optionally also try zero-padding (replace segment with zeros) as
    an alternative to NUN substitution.
"""
from __future__ import annotations

import numpy as np
from config import Config, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Shannon entropy helper
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(values: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Shannon entropy of a 1-D signal segment.

    Uses histogram-based discretisation into n_bins equal-width bins.
    Returns entropy in bits (log base 2).
    """
    if len(values) < 2:
        return 0.0
    counts, _ = np.histogram(values, bins=n_bins)
    probs = counts / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


# ─────────────────────────────────────────────────────────────────────────────
# AB-CF core
# ─────────────────────────────────────────────────────────────────────────────

class ABCF:
    """
    Attention-Based Counterfactual (AB-CF) for multivariate time series.

    Parameters
    ----------
    model        : classifier with predict_proba(X[None]) → (1, K) and
                   predict(X) → int
    X_train      : (N_tr, T, D) training data
    y_train      : (N_tr,) training labels
    n_segments   : number of equal-length segments to divide T into
    n_bins       : histogram bins for Shannon entropy computation
    window_frac  : minimum segment length as fraction of T
    max_swaps    : maximum (channel, segment) swaps before giving up
    """

    def __init__(self,
                 model,
                 X_train:    np.ndarray,
                 y_train:    np.ndarray,
                 n_segments: int   = 5,
                 n_bins:     int   = 10,
                 window_frac: float = 0.15,
                 max_swaps:  int   = 50):
        self.model       = model
        self.X_train     = X_train
        self.y_train     = y_train
        self.n_segments  = n_segments
        self.n_bins      = n_bins
        self.window_frac = window_frac
        self.max_swaps   = max_swaps

    # ── NUN search ────────────────────────────────────────────────────────────

    def _find_nun(self, X: np.ndarray, y_tgt: int) -> np.ndarray:
        """
        Nearest Unlike Neighbor: closest training sample of class y_tgt.
        X : (T, D) → returns (T, D)
        """
        mask    = self.y_train == y_tgt
        X_tgt   = self.X_train[mask]               # (M, T, D)
        X_flat  = X.flatten()
        dists   = np.linalg.norm(
            X_tgt.reshape(len(X_tgt), -1) - X_flat, axis=1
        )
        return X_tgt[np.argmin(dists)]              # (T, D)

    # ── Segment boundaries ────────────────────────────────────────────────────

    def _get_segments(self, T: int) -> list[tuple[int, int]]:
        """
        Compute segment (start, end) index pairs.

        Ensures each segment has at least ceil(window_frac * T) timesteps.
        """
        min_len  = max(1, int(np.ceil(self.window_frac * T)))
        n_seg    = min(self.n_segments, T // min_len)
        n_seg    = max(1, n_seg)

        seg_len  = T // n_seg
        segments = []
        for k in range(n_seg):
            start = k * seg_len
            end   = start + seg_len if k < n_seg - 1 else T
            segments.append((start, end))
        return segments

    # ── Entropy ranking ───────────────────────────────────────────────────────

    def _rank_segments(self,
                       X:        np.ndarray,
                       segments: list[tuple[int, int]]) -> list[tuple[float, int, int, int, int]]:
        """
        Rank all (channel, segment) pairs by Shannon entropy of X (descending).

        Returns sorted list of (entropy, channel_idx, seg_idx, start, end).
        """
        T, D   = X.shape
        ranked = []

        for s_idx, (start, end) in enumerate(segments):
            for d in range(D):
                seg_vals = X[start:end, d]
                H        = _shannon_entropy(seg_vals, self.n_bins)
                ranked.append((H, d, s_idx, start, end))

        # Sort by entropy descending — high entropy first
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    # ── Core explain ──────────────────────────────────────────────────────────

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
        info   : dict with full explanation metadata
        """
        T, D = X.shape

        # Step 1: Find NUN of target class
        NUN = self._find_nun(X, y_tgt)

        # Step 2: Get segment boundaries
        segments = self._get_segments(T)

        # Step 3: Rank (channel, segment) pairs by entropy
        ranked = self._rank_segments(X, segments)

        # Step 4: Incremental greedy swap
        X_cf        = X.copy()
        swapped     = []           # list of (channel, seg_idx, start, end)
        entropy_map = np.zeros((D, len(segments)))   # for visualization
        prob_trace  = []

        # Fill entropy map for all pairs
        for H, d, s_idx, start, end in ranked:
            entropy_map[d, s_idx] = H

        n_swaps = 0
        for H, d, s_idx, start, end in ranked:
            if n_swaps >= self.max_swaps:
                break

            # Check validity before swap
            proba = self.model.predict_proba(X_cf[None])[0]
            prob_trace.append(float(proba[y_tgt]))

            if np.argmax(proba) == y_tgt:
                break

            # Swap segment [start:end] of channel d from NUN
            X_cf[start:end, d] = NUN[start:end, d]
            swapped.append((d, s_idx, start, end, H))
            n_swaps += 1

        # Final evaluation
        final_proba = self.model.predict_proba(X_cf[None])[0]
        pred_class  = int(np.argmax(final_proba))
        valid       = (pred_class == y_tgt)

        # Sparsity: fraction of (timestep, channel) elements unchanged
        delta     = X_cf - X
        n_changed = float(np.sum(np.abs(delta) > 1e-8))
        sparsity  = 1.0 - (n_changed / (T * D))

        # Compactness: fraction of channels with any change
        chan_changed = float(np.sum(np.any(np.abs(delta) > 1e-8, axis=0)))
        compactness  = 1.0 - (chan_changed / D)

        # Channel-level importance: sum of entropy of swapped segments per channel
        channel_importance = np.zeros(D)
        for d, s_idx, start, end, H in swapped:
            channel_importance[d] += H

        # Segment-level importance: 1 if any channel was swapped in that segment
        segment_importance = np.zeros(len(segments))
        for d, s_idx, start, end, H in swapped:
            segment_importance[s_idx] += 1.0

        info = {
            "NUN"                : NUN,
            "segments"           : segments,
            "swapped"            : swapped,
            "n_swaps"            : n_swaps,
            "entropy_map"        : entropy_map,       # (D, n_seg)
            "channel_importance" : channel_importance, # (D,)
            "segment_importance" : segment_importance, # (n_seg,)
            "prob_trace"         : prob_trace,
            "final_proba"        : final_proba,
            "pred_class"         : pred_class,
            "validity"           : valid,
            "sparsity"           : sparsity,
            "compactness"        : compactness,
            "orig_class"         : int(self.model.predict(X)),
            "cf_class"           : pred_class,
            "delta"              : delta,
        }

        return X_cf, info


# ─────────────────────────────────────────────────────────────────────────────
# Zero-padding variant
# ─────────────────────────────────────────────────────────────────────────────

class ABCFZeroPad(ABCF):
    """
    AB-CF variant that replaces segments with zeros instead of NUN values.

    Useful when no NUN is available or to test the contribution of the
    NUN substitution vs. pure zeroing.
    """

    def explain(self, X: np.ndarray, y_tgt: int) -> tuple[np.ndarray, dict]:
        T, D     = X.shape
        NUN      = self._find_nun(X, y_tgt)
        segments = self._get_segments(T)
        ranked   = self._rank_segments(X, segments)

        X_cf       = X.copy()
        swapped    = []
        entropy_map = np.zeros((D, len(segments)))
        prob_trace = []

        for H, d, s_idx, start, end in ranked:
            entropy_map[d, s_idx] = H

        n_swaps = 0
        for H, d, s_idx, start, end in ranked:
            if n_swaps >= self.max_swaps:
                break

            proba = self.model.predict_proba(X_cf[None])[0]
            prob_trace.append(float(proba[y_tgt]))

            if np.argmax(proba) == y_tgt:
                break

            # Zero-pad instead of NUN substitution
            X_cf[start:end, d] = 0.0
            swapped.append((d, s_idx, start, end, H))
            n_swaps += 1

        final_proba = self.model.predict_proba(X_cf[None])[0]
        pred_class  = int(np.argmax(final_proba))
        valid       = (pred_class == y_tgt)

        delta        = X_cf - X
        n_changed    = float(np.sum(np.abs(delta) > 1e-8))
        sparsity     = 1.0 - (n_changed / (T * D))
        chan_changed  = float(np.sum(np.any(np.abs(delta) > 1e-8, axis=0)))
        compactness   = 1.0 - (chan_changed / D)

        channel_importance = np.zeros(D)
        for d, s_idx, start, end, H in swapped:
            channel_importance[d] += H

        segment_importance = np.zeros(len(segments))
        for d, s_idx, start, end, H in swapped:
            segment_importance[s_idx] += 1.0

        info = {
            "NUN"                : NUN,
            "segments"           : segments,
            "swapped"            : swapped,
            "n_swaps"            : n_swaps,
            "entropy_map"        : entropy_map,
            "channel_importance" : channel_importance,
            "segment_importance" : segment_importance,
            "prob_trace"         : prob_trace,
            "final_proba"        : final_proba,
            "pred_class"         : pred_class,
            "validity"           : valid,
            "sparsity"           : sparsity,
            "compactness"        : compactness,
            "orig_class"         : int(self.model.predict(X)),
            "cf_class"           : pred_class,
            "delta"              : delta,
            "variant"            : "zero_pad",
        }

        return X_cf, info
