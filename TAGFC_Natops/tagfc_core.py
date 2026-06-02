"""
tagfc_core.py — TAGFC: Transformer Attention-Guided Frequency Counterfactual.

Applied to NATOPS gesture classification (T=51, D=24, K=6).

Algorithm (5 steps):
─────────────────────────────────────────────────────────────────────────────
1. ATTENTION ROLLOUT  →  temporal saliency s ∈ [0,1]^T
   Â^l = 0.5·A^l + 0.5·I  (per layer, averaged over heads)
   Propagate: R = Â^1 @ Â^2 @ ... @ Â^L
   s = column_sum(R), normalised to [0,1]

2. HAAR DWT  →  coefficient space C ∈ R^{D×L}   (T=51 → L=64)
   Separate gesture motion into frequency bands:
     Approx (slow overall motion) | Detail levels (fast joint transitions)

3. OMEGA WEIGHTS  →  per-coefficient penalty ω ∈ [0,1]^{D×L}
   G = |∂logit_{y_tgt}/∂X|  (input gradient, Gaussian-smoothed)
   G_wav = |DWT(G)|  (gradient in wavelet domain)
   M = s_coeff[None,:] × G_wav  +  ε    (joint attention × gradient importance)
   ω = 1/M  normalised to [0,1]
   High ω = unimportant region → cheap to perturb

4. 4-TERM OBJECTIVE
   L_total = L_flip  +  λ1·L_sparse  +  λ2·L_cross  +  λ3·L_manifold
   L_flip     = -log P(y_tgt | X_cf)           (validity)
   L_sparse   = Σ ω·|δ|                        (ω-weighted L1 sparsity)
   L_cross    = (X_cf-X)_mean^T · Σ^{-1} · (X_cf-X)_mean  (inter-sensor coherence)
   L_manifold = ||X_cf - X_nn||^2 / (T·D)     (plausibility)

5. PROXIMAL GRADIENT DESCENT
   δ ← δ - η · ∇_δ (L_flip + λ2·L_cross + λ3·L_manifold)
   δ ← prox_{λ1·η·ω}(δ)   [soft-thresholding for exact L1 sparsity]
   δ ← clip(δ, -Δ, Δ)
   X_cf = iDWT(C + δ)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from config import Config, cfg
from haar_wavelet import HaarWT


def compute_omega(X: np.ndarray,
                  model,
                  target_class: int,
                  haar: HaarWT,
                  grad_smooth_sigma: float = 1.5) -> np.ndarray:
    """
    Compute per-coefficient penalty weights ω ∈ [0,1]^{D×L}.

    Small ω = highly important coefficient (model penalises changes here).
    Large ω = unimportant coefficient (cheap to perturb).

    Parameters
    ----------
    X                : (T, D) input gesture sequence
    model            : TransformerNATOPS
    target_class     : int  target gesture class
    haar             : HaarWT instance
    grad_smooth_sigma: σ for Gaussian smoothing of gradient magnitude

    Returns
    -------
    omega : (D, L) float32 in [0, 1]
    """
    T, D = X.shape

    # Step 1: Attention rollout → temporal saliency (T,)
    s = model.attention_rollout(X)

    # Step 2: Input gradient → (T, D), smooth, normalise
    G_raw = model.input_gradient(X, target_class)
    G     = gaussian_filter1d(np.abs(G_raw), sigma=grad_smooth_sigma, axis=0)
    G     = G / (G.max() + 1e-8)

    # Step 3: Gradient in wavelet domain → (D, L)
    G_wav = haar.dwt2d(G)
    G_wav = np.abs(G_wav) / (np.abs(G_wav).max() + 1e-8)

    # Step 4: Map temporal saliency to coefficient positions
    L     = G_wav.shape[1]
    idx   = np.arange(L)
    t_pos = np.round(idx / L * T).astype(int).clip(0, T - 1)
    s_coeff = s[t_pos]              # (L,) saliency at each coefficient position

    # Step 5: Joint importance M = saliency × gradient_wavelet
    M     = s_coeff[None, :] * G_wav + 1e-8   # (D, L)
    omega = 1.0 / M
    omega = omega / (omega.max() + 1e-8)       # normalise to [0, 1]

    return omega.astype(np.float32)


class TAGFCOptimizer:
    """
    TAGFC counterfactual generator for NATOPS gesture sequences.

    Generates X_cf by minimising:
        L_flip  +  λ1·L_sparse  +  λ2·L_cross  +  λ3·L_manifold
    via proximal gradient descent in Haar wavelet coefficient space.

    Parameters
    ----------
    model   : TransformerNATOPS (white-box access required)
    X_train : (N_tr, T, D) training data
    y_train : (N_tr,) training labels
    c       : Config
    """

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray,
                 c: Config = cfg):
        self.model   = model
        self.X_train = X_train
        self.y_train = y_train
        self.c       = c
        self.haar    = HaarWT(levels=c.WAV_LEVELS)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        print("[TAGFC] Computing training covariance matrix ...")
        self.Sigma_inv = self._compute_sigma_inv()

    def _compute_sigma_inv(self) -> np.ndarray:
        """Inverse covariance of training data (for Mahalanobis L_cross)."""
        X_flat = self.X_train.reshape(-1, self.X_train.shape[-1]).astype(np.float64)
        Sigma  = np.cov(X_flat.T) + 1e-4 * np.eye(X_flat.shape[1])
        cond   = np.linalg.cond(Sigma)
        print(f"[TAGFC] Σ condition number = {cond:.2f}")
        return np.linalg.inv(Sigma).astype(np.float32)

    def _find_nn(self, X: np.ndarray, y_tgt: int) -> np.ndarray:
        """Nearest training sample of target class y_tgt."""
        mask  = self.y_train == y_tgt
        X_tgt = self.X_train[mask]
        diffs = (X_tgt - X[None]).reshape(len(X_tgt), -1)
        dists = np.linalg.norm(diffs, axis=1)
        return X_tgt[dists.argmin()]

    def _flip_gradient(self, X_cf: np.ndarray, y_tgt: int) -> np.ndarray:
        """∂L_flip/∂X_cf  where  L_flip = -log P(y_tgt | X_cf)."""
        x_t    = torch.tensor(X_cf[None], dtype=torch.float32, requires_grad=True)
        logits = self.model(x_t)
        probs  = torch.softmax(logits, dim=-1)
        loss   = -torch.log(probs[0, y_tgt] + 1e-8)
        loss.backward()
        return x_t.grad[0].detach().cpu().numpy()    # (T, D)

    def optimize(self, X: np.ndarray, y_tgt: int,
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate a counterfactual for query gesture X targeting class y_tgt.

        Parameters
        ----------
        X       : (T, D) query gesture sequence
        y_tgt   : int  target gesture class
        verbose : print iteration log every 50 steps

        Returns
        -------
        X_cf    : (T, D) counterfactual gesture sequence
        delta   : (D, L) wavelet-coefficient perturbation
        info    : dict  — explanation artefacts + optimisation history
        """
        T, D = X.shape
        c    = self.c

        # DWT of query
        C = self.haar.dwt2d(X)   # (D, L)
        L = C.shape[1]

        # Nearest target-class sample (for L_manifold)
        X_nn = self._find_nn(X, y_tgt)

        # Omega weights and temporal saliency
        omega      = compute_omega(X, self.model, y_tgt, self.haar, c.GRAD_SMOOTH_SIGMA)
        s_temporal = self.model.attention_rollout(X)
        orig_class = self.model.predict(X)

        # Initialise delta
        delta        = np.zeros((D, L), dtype=np.float32)
        best_delta   = delta.copy()
        best_loss    = float("inf")
        patience     = 0
        ever_flipped = False
        history      = []

        for k in range(c.MAX_ITER):
            # Reconstruct CF from current delta
            X_cf = self.haar.idwt2d(C + delta, original_length=T)

            # ── Cross-sensor coherence term ──────────────────────────────────
            diff       = (X_cf - X).mean(axis=0)           # (D,)
            L_cross    = float(diff @ self.Sigma_inv @ diff)

            # ── Manifold plausibility term ───────────────────────────────────
            L_manifold = float(np.sum((X_cf - X_nn) ** 2) / (T * D))

            # ── Gradients w.r.t. X_cf ────────────────────────────────────────
            grad_flip     = self._flip_gradient(X_cf, y_tgt)           # (T, D)
            grad_cross    = (self.Sigma_inv @ diff) / T                 # (D,) → broadcast
            grad_manifold = 2.0 * (X_cf - X_nn) / (T * D)              # (T, D)

            grad_X = (grad_flip
                      + c.LAMBDA_CROSS    * grad_cross[None, :]
                      + c.LAMBDA_MANIFOLD * grad_manifold)              # (T, D)

            # ── Convert gradient to wavelet domain ───────────────────────────
            grad_delta = self.haar.dwt2d(grad_X)             # (D, L)

            # ── Gradient step ────────────────────────────────────────────────
            delta_new = delta - c.LR_OPT * grad_delta

            # ── Proximal operator: soft-threshold for exact L1 sparsity ──────
            threshold  = c.LR_OPT * c.LAMBDA_SPARSE * omega  # (D, L)
            delta_prox = np.sign(delta_new) * np.maximum(
                np.abs(delta_new) - threshold, 0.0
            )

            # ── Bound constraint ──────────────────────────────────────────────
            delta = np.clip(delta_prox, -c.DELTA_BOUND, c.DELTA_BOUND)

            # ── Evaluate ──────────────────────────────────────────────────────
            X_cf_check  = self.haar.idwt2d(C + delta, original_length=T)
            probs       = self.model.predict_proba(X_cf_check)
            pred        = int(probs.argmax())
            flipped     = (pred == y_tgt)

            L_flip   = float(-np.log(probs[y_tgt] + 1e-8))
            L_sparse = float(np.sum(omega * np.abs(delta)))
            total    = (L_flip
                        + c.LAMBDA_SPARSE   * L_sparse
                        + c.LAMBDA_CROSS    * L_cross
                        + c.LAMBDA_MANIFOLD * L_manifold)

            # ── Track best ───────────────────────────────────────────────────
            if flipped:
                patience += 1
                if not ever_flipped or total < best_loss:
                    best_delta = delta.copy()
                    best_loss  = total
                ever_flipped = True
                if patience >= c.PATIENCE:
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

        # ── Final CF ──────────────────────────────────────────────────────────
        X_cf_final  = self.haar.idwt2d(C + best_delta, original_length=T)
        final_probs = self.model.predict_proba(X_cf_final)
        final_pred  = int(final_probs.argmax())

        # ── Explanation artefacts ──────────────────────────────────────────────
        sensor_imp = np.abs(best_delta).sum(axis=1)   # (D,)  per-channel importance
        freq_imp   = np.abs(best_delta).sum(axis=0)   # (L,)  per-frequency importance

        info = {
            "validity":          final_pred == y_tgt,
            "orig_class":        orig_class,
            "cf_class":          final_pred,
            "cf_confidence":     float(final_probs[y_tgt]),
            "sensor_importance": sensor_imp,        # (D,)
            "temporal_saliency": s_temporal,        # (T,)
            "freq_importance":   freq_imp,          # (L,)
            "delta":             best_delta,        # (D, L) wavelet delta
            "omega":             omega,             # (D, L) penalty weights
            "history":           history,
        }

        return X_cf_final, best_delta, info
