"""
tagfc_core.py — TAGFC applied to NATOPS gesture classification.

Same 5-step algorithm as CMAPSS/AQI variants:
  1. Attention Rollout  → temporal saliency s ∈ [0,1]^T  (T=51)
  2. Haar DWT           → coefficient space C ∈ R^{D×L}  (D=24, L=64)
  3. Omega weights      → per-coefficient penalty ω ∈ [0,1]^{D×L}
  4. 4-term objective   → L_flip + λ1·L_sparse + λ2·L_cross + λ3·L_manifold
  5. Proximal GD        → soft-threshold + clip delta
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
    Per-coefficient penalty weights omega ∈ [0,1]^{D×L}.
    Small omega = important (cheap to perturb).
    """
    T, D = X.shape

    s     = model.attention_rollout(X)
    G_raw = model.input_gradient(X, target_class)
    G     = gaussian_filter1d(np.abs(G_raw), sigma=grad_smooth_sigma, axis=0)
    G     = G / (G.max() + 1e-8)

    G_wav = haar.dwt2d(G)
    G_wav = np.abs(G_wav) / (np.abs(G_wav).max() + 1e-8)

    L       = G_wav.shape[1]
    idx     = np.arange(L)
    t_pos   = np.round(idx / L * T).astype(int).clip(0, T - 1)
    s_coeff = s[t_pos]

    M     = s_coeff[None, :] * G_wav + 1e-8
    omega = 1.0 / M
    omega = omega / (omega.max() + 1e-8)
    return omega.astype(np.float32)


class TAGFCOptimizer:
    """
    Generates a counterfactual for a single NATOPS gesture query X
    by minimising:
        L_flip  +  λ1·L_sparse  +  λ2·L_cross  +  λ3·L_manifold
    via proximal gradient descent in Haar wavelet coefficient space.
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

        self.Sigma_inv = self._compute_sigma_inv()

    def _compute_sigma_inv(self) -> np.ndarray:
        X_flat = self.X_train.reshape(-1, self.X_train.shape[-1]).astype(np.float64)
        Sigma  = np.cov(X_flat.T) + 1e-4 * np.eye(X_flat.shape[1])
        cond   = np.linalg.cond(Sigma)
        print(f"[TAGFC] Σ condition number = {cond:.2f}")
        return np.linalg.inv(Sigma).astype(np.float32)

    def _find_nn(self, X: np.ndarray, y_tgt: int) -> np.ndarray:
        mask  = self.y_train == y_tgt
        X_tgt = self.X_train[mask]
        diffs = (X_tgt - X[None]).reshape(len(X_tgt), -1)
        dists = np.linalg.norm(diffs, axis=1)
        return X_tgt[dists.argmin()]

    def _flip_gradient(self, X_cf: np.ndarray, y_tgt: int) -> np.ndarray:
        x_t    = torch.tensor(X_cf[None], dtype=torch.float32, requires_grad=True)
        logits = self.model(x_t)
        probs  = torch.softmax(logits, dim=-1)
        loss   = -torch.log(probs[0, y_tgt] + 1e-8)
        loss.backward()
        return x_t.grad[0].detach().cpu().numpy()

    def optimize(self, X: np.ndarray, y_tgt: int,
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate a counterfactual for query X targeting gesture class y_tgt.

        Returns
        -------
        X_cf  : (T, D) counterfactual gesture sequence
        delta : (D, L) wavelet-coefficient perturbation
        info  : dict with explanation dimensions and optimisation history
        """
        T, D = X.shape
        c    = self.c

        C   = self.haar.dwt2d(X)
        L   = C.shape[1]

        X_nn       = self._find_nn(X, y_tgt)
        omega      = compute_omega(X, self.model, y_tgt, self.haar, c.GRAD_SMOOTH_SIGMA)
        s_temporal = self.model.attention_rollout(X)
        orig_class = self.model.predict(X)

        delta        = np.zeros((D, L), dtype=np.float32)
        best_delta   = delta.copy()
        best_loss    = float("inf")
        patience     = 0
        ever_flipped = False
        history      = []

        for k in range(c.MAX_ITER):
            X_cf = self.haar.idwt2d(C + delta, original_length=T)

            diff       = (X_cf - X).mean(axis=0)
            L_cross    = float(diff @ self.Sigma_inv @ diff)
            L_manifold = float(np.sum((X_cf - X_nn) ** 2) / (T * D))

            grad_flip     = self._flip_gradient(X_cf, y_tgt)
            grad_cross    = (self.Sigma_inv @ diff) / T
            grad_manifold = 2.0 * (X_cf - X_nn) / (T * D)

            grad_X = (grad_flip
                      + c.LAMBDA_CROSS    * grad_cross[None, :]
                      + c.LAMBDA_MANIFOLD * grad_manifold)

            grad_delta = self.haar.dwt2d(grad_X)
            delta_new  = delta - c.LR_OPT * grad_delta
            threshold  = c.LR_OPT * c.LAMBDA_SPARSE * omega
            delta_prox = np.sign(delta_new) * np.maximum(
                np.abs(delta_new) - threshold, 0.0
            )
            delta = np.clip(delta_prox, -c.DELTA_BOUND, c.DELTA_BOUND)

            X_cf_check = self.haar.idwt2d(C + delta, original_length=T)
            probs      = self.model.predict_proba(X_cf_check)
            pred       = int(probs.argmax())
            flipped    = (pred == y_tgt)

            L_flip   = float(-np.log(probs[y_tgt] + 1e-8))
            L_sparse = float(np.sum(omega * np.abs(delta)))
            total    = (L_flip
                        + c.LAMBDA_SPARSE   * L_sparse
                        + c.LAMBDA_CROSS    * L_cross
                        + c.LAMBDA_MANIFOLD * L_manifold)

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

        X_cf_final  = self.haar.idwt2d(C + best_delta, original_length=T)
        final_probs = self.model.predict_proba(X_cf_final)
        final_pred  = int(final_probs.argmax())

        sensor_imp = np.abs(best_delta).sum(axis=1)
        freq_imp   = np.abs(best_delta).sum(axis=0)

        info = {
            "flipped":           final_pred == y_tgt,
            "orig_class":        orig_class,
            "cf_class":          final_pred,
            "cf_confidence":     float(final_probs[y_tgt]),
            "sensor_importance": sensor_imp,
            "temporal_saliency": s_temporal,
            "freq_importance":   freq_imp,
            "delta":             best_delta,
            "omega":             omega,
            "history":           history,
        }
        return X_cf_final, best_delta, info
