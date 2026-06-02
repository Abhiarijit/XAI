"""
tagfc_core.py — TAGFC: Transformer Attention-Guided Frequency Counterfactual

Implements:
  1. compute_omega()   — per-coefficient importance weights
  2. TAGFCOptimizer    — proximal gradient descent loop
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from config import Config, cfg
from haar_wavelet import HaarWT


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Importance weights omega
# ─────────────────────────────────────────────────────────────────────────────

def compute_omega(X: np.ndarray,
                  model,
                  target_class: int,
                  haar: HaarWT,
                  grad_smooth_sigma: float = 1.5) -> np.ndarray:
    """
    Compute per-coefficient penalty weights omega ∈ [0, 1]^{D × L}.

    Small omega → important coefficient (cheap to change — TAGFC targets it).
    Large omega → unimportant coefficient (expensive — TAGFC avoids it).

    Algorithm
    ---------
    1. Attention rollout saliency  s ∈ [0,1]^T
    2. |Input gradient| smoothed   G ∈ [0,1]^{T×D}
    3. Wavelet-space gradient      G_wav = DWT(G) ∈ R^{D×L}
    4. Map saliency to coeff positions  s_coeff ∈ [0,1]^L
    5. M = s_coeff * G_wav + eps
    6. omega = 1/M, normalised to [0,1]

    Parameters
    ----------
    X              : (T, D) normalised query window
    model          : TransformerFD001 (eval mode)
    target_class   : int
    haar           : HaarWT instance
    grad_smooth_sigma : Gaussian smoothing σ along time axis

    Returns
    -------
    omega : (D, L) float32
    """
    T, D = X.shape

    # Step 1: attention rollout saliency
    s = model.attention_rollout(X)                    # (T,)

    # Step 2: input gradient (absolute, smoothed)
    G_raw = model.input_gradient(X, target_class)     # (T, D)
    G     = gaussian_filter1d(np.abs(G_raw), sigma=grad_smooth_sigma, axis=0)
    G     = G / (G.max() + 1e-8)                      # (T, D)

    # Step 3: gradient in wavelet space via DWT
    G_wav = haar.dwt2d(G)                             # (D, L)
    G_wav = np.abs(G_wav)
    G_wav = G_wav / (G_wav.max() + 1e-8)

    L = G_wav.shape[1]

    # Step 4: map temporal saliency → coefficient positions
    idx     = np.arange(L)
    t_pos   = np.round(idx / L * T).astype(int).clip(0, T - 1)
    s_coeff = s[t_pos]                                # (L,)

    # Step 5: importance M
    M = s_coeff[None, :] * G_wav + 1e-8              # (D, L)

    # Step 6: omega = 1/M, normalised
    omega = 1.0 / M
    omega = omega / (omega.max() + 1e-8)
    return omega.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Steps 4 & 5 — TAGFC Proximal Gradient Descent
# ─────────────────────────────────────────────────────────────────────────────

class TAGFCOptimizer:
    """
    Generates a counterfactual explanation for a single query instance X
    by minimising:

        L_flip  +  λ₁·L_sparse  +  λ₂·L_cross  +  λ₃·L_manifold

    via proximal gradient descent in Haar wavelet coefficient space.

    Parameters
    ----------
    model    : TransformerFD001 (will be set to eval mode, no grad on params)
    X_train  : (N, T, D) normalised training windows
    y_train  : (N,)      integer class labels
    c        : Config instance
    """

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray,
                 c: Config = cfg):
        self.model   = model
        self.X_train = X_train
        self.y_train = y_train
        self.c       = c
        self.haar    = HaarWT(levels=c.WAV_LEVELS)

        # Freeze model parameters for the optimisation phase
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Pre-compute Σ⁻¹ once (Mahalanobis coherence term)
        self.Sigma_inv = self._compute_sigma_inv()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_sigma_inv(self) -> np.ndarray:
        X_flat = self.X_train.reshape(-1, self.X_train.shape[-1]).astype(np.float64)
        Sigma  = np.cov(X_flat.T) + 1e-4 * np.eye(X_flat.shape[1])
        cond   = np.linalg.cond(Sigma)
        print(f"[TAGFC] Σ condition number = {cond:.2f}")
        return np.linalg.inv(Sigma).astype(np.float32)

    def _find_nn(self, X: np.ndarray, y_tgt: int) -> np.ndarray:
        """Nearest training sample of class y_tgt (Euclidean in flattened space)."""
        mask   = self.y_train == y_tgt
        X_tgt  = self.X_train[mask]
        diffs  = (X_tgt - X[None]).reshape(len(X_tgt), -1)
        dists  = np.linalg.norm(diffs, axis=1)
        return X_tgt[dists.argmin()]

    def _flip_gradient(self, X_cf: np.ndarray, y_tgt: int) -> np.ndarray:
        """
        Gradient of  −log P(y_tgt | X_cf)  w.r.t. X_cf  via PyTorch autograd.
        X_cf : (T, D) numpy  →  returns (T, D) numpy
        """
        x_t    = torch.tensor(X_cf[None], dtype=torch.float32, requires_grad=True)
        logits = self.model(x_t)
        probs  = torch.softmax(logits, dim=-1)
        loss   = -torch.log(probs[0, y_tgt] + 1e-8)
        loss.backward()
        return x_t.grad[0].detach().cpu().numpy()

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize(self, X: np.ndarray, y_tgt: int,
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate a counterfactual for query X targeting class y_tgt.

        Parameters
        ----------
        X       : (T, D) normalised single window
        y_tgt   : int   target class
        verbose : print progress every 50 iterations

        Returns
        -------
        X_cf    : (T, D) counterfactual window
        delta   : (D, L) wavelet-coefficient perturbation
        info    : dict with explanation dimensions and optimisation history
        """
        T, D = X.shape
        c    = self.c

        # ── Step 2: Haar DWT of original signal ──────────────────────────────
        C = self.haar.dwt2d(X)    # (D, L)
        L = C.shape[1]

        # ── Preprocessing ─────────────────────────────────────────────────────
        X_nn         = self._find_nn(X, y_tgt)           # (T, D)
        omega        = compute_omega(X, self.model, y_tgt, self.haar, c.GRAD_SMOOTH_SIGMA)
        s_temporal   = self.model.attention_rollout(X)   # (T,) for explanation
        orig_class   = self.model.predict(X)

        # ── Initialise delta ──────────────────────────────────────────────────
        delta      = np.zeros((D, L), dtype=np.float32)
        best_delta = delta.copy()
        best_loss  = float("inf")
        patience   = 0
        ever_flipped = False
        history    = []

        # ── Proximal GD loop ──────────────────────────────────────────────────
        for k in range(c.MAX_ITER):

            # A — reconstruct counterfactual
            X_cf = self.haar.idwt2d(C + delta, original_length=T)

            # B — analytical smooth-term losses
            diff       = (X_cf - X).mean(axis=0)                          # (D,)
            L_cross    = float(diff @ self.Sigma_inv @ diff)
            L_manifold = float(np.sum((X_cf - X_nn) ** 2) / (T * D))

            # C — gradient of flip loss (PyTorch autograd)
            grad_flip = self._flip_gradient(X_cf, y_tgt)                  # (T, D)

            # C — analytical gradients of smooth terms
            grad_cross    = (self.Sigma_inv @ diff) / T                   # (D,) broadcast
            grad_manifold = 2.0 * (X_cf - X_nn) / (T * D)                # (T, D)

            grad_X = (grad_flip
                      + c.LAMBDA_CROSS    * grad_cross[None, :]
                      + c.LAMBDA_MANIFOLD * grad_manifold)                # (T, D)

            # D — chain through iDWT adjoint (= DWT for orthogonal Haar)
            grad_delta = self.haar.dwt2d(grad_X)                          # (D, L)

            # E — gradient step
            delta_new = delta - c.LR_OPT * grad_delta

            # F — proximal step: soft-threshold for L1 (L_sparse)
            threshold  = c.LR_OPT * c.LAMBDA_SPARSE * omega              # (D, L)
            delta_prox = np.sign(delta_new) * np.maximum(
                np.abs(delta_new) - threshold, 0.0
            )

            # G — clip
            delta = np.clip(delta_prox, -c.DELTA_BOUND, c.DELTA_BOUND)

            # H — check flip
            X_cf_check  = self.haar.idwt2d(C + delta, original_length=T)
            probs       = self.model.predict_proba(X_cf_check)
            pred        = int(probs.argmax())
            flipped     = (pred == y_tgt)

            L_flip    = float(-np.log(probs[y_tgt] + 1e-8))
            L_sparse  = float(np.sum(omega * np.abs(delta)))
            total     = (L_flip
                         + c.LAMBDA_SPARSE   * L_sparse
                         + c.LAMBDA_CROSS    * L_cross
                         + c.LAMBDA_MANIFOLD * L_manifold)

            if flipped:
                patience += 1
                if not ever_flipped or total < best_loss:
                    best_delta   = delta.copy()
                    best_loss    = total
                ever_flipped = True
                if patience >= c.PATIENCE:
                    if verbose:
                        print(f"  [TAGFC] Converged at iter {k}")
                    break
            else:
                patience = 0
                if not ever_flipped:
                    best_delta = delta.copy()

            history.append({
                "iter": k, "L_flip": L_flip, "L_sparse": L_sparse,
                "L_cross": L_cross, "L_manifold": L_manifold,
                "total": total, "flipped": flipped,
                "cf_conf": float(probs[y_tgt]),
            })

            if verbose and k % 50 == 0:
                print(f"  Iter {k:4d} | flip={L_flip:.3f} sparse={L_sparse:.3f} "
                      f"cross={L_cross:.3f} mfld={L_manifold:.3f} "
                      f"conf={probs[y_tgt]:.3f} pred={pred}")

        # ── Final counterfactual ───────────────────────────────────────────────
        X_cf_final  = self.haar.idwt2d(C + best_delta, original_length=T)
        final_probs = self.model.predict_proba(X_cf_final)
        final_pred  = int(final_probs.argmax())

        # ── 3-D explanation ───────────────────────────────────────────────────
        sensor_imp = np.abs(best_delta).sum(axis=1)   # (D,) sensor importance
        freq_imp   = np.abs(best_delta).sum(axis=0)   # (L,) frequency importance

        info = {
            "flipped":          final_pred == y_tgt,
            "orig_class":       orig_class,
            "cf_class":         final_pred,
            "cf_confidence":    float(final_probs[y_tgt]),
            "sensor_importance": sensor_imp,
            "temporal_saliency": s_temporal,
            "freq_importance":   freq_imp,
            "delta":             best_delta,
            "omega":             omega,
            "history":           history,
        }
        return X_cf_final, best_delta, info
