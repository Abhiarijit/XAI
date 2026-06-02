"""
haar_wavelet.py — Pure-numpy Haar Discrete Wavelet Transform.

Signals are zero-padded to the next power-of-2 length so the DWT is
exactly invertible (iDWT ∘ DWT = identity, up to float precision).

Coefficient layout for a length-L = 2^k output with `levels` stages:
    [ approx_L  |  detail_L  |  detail_{L-1}  |  ...  |  detail_1 ]
    sizes:  n_a       n_a        2·n_a                   L/2
    where n_a = L / 2^levels

For T=14, levels=3:  L = 16,  n_a = 2
    [  approx(2) | detail_L3(2) | detail_L2(4) | detail_L1(8) ]  total=16
"""
from __future__ import annotations
import numpy as np


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class HaarWT:
    """Orthogonal Haar Wavelet Transform over 1-D and 2-D (multivariate) arrays."""

    def __init__(self, levels: int = 3):
        self.levels = levels

    # ── 1-D ──────────────────────────────────────────────────────────────────

    def dwt(self, x: np.ndarray) -> np.ndarray:
        """
        x       : (T,) 1-D signal
        returns : (L,) coefficient vector,  L = next_pow2(T)
        """
        T = len(x)
        L = _next_pow2(T)

        a = np.empty(L, dtype=np.float64)
        a[:T] = x
        if L > T:
            a[T:] = x[-1]

        details: list[np.ndarray] = []
        for _ in range(self.levels):
            approx = (a[0::2] + a[1::2]) / np.sqrt(2)
            detail = (a[0::2] - a[1::2]) / np.sqrt(2)
            details.append(detail)
            a = approx

        return np.concatenate([a] + details[::-1]).astype(np.float32)

    def idwt(self, coeffs: np.ndarray, original_length: int | None = None) -> np.ndarray:
        """
        coeffs          : (L,) output of dwt()
        original_length : if given, truncate reconstruction to this length
        returns         : (L,) or (original_length,) reconstructed signal
        """
        c = coeffs.astype(np.float64)
        L = len(c)
        n_approx = L >> self.levels

        a = c[:n_approx].copy()
        pos      = n_approx
        n_detail = n_approx

        for _ in range(self.levels):
            d = c[pos : pos + n_detail]
            pos      += n_detail
            n_detail *= 2

            rec = np.empty(2 * len(a), dtype=np.float64)
            rec[0::2] = (a + d) / np.sqrt(2)
            rec[1::2] = (a - d) / np.sqrt(2)
            a = rec

        if original_length is not None:
            return a[:original_length].astype(np.float32)
        return a.astype(np.float32)

    def coeff_len(self, T: int) -> int:
        """Number of DWT coefficients for a signal of length T."""
        return _next_pow2(T)

    # ── 2-D (multivariate) ───────────────────────────────────────────────────

    def dwt2d(self, X: np.ndarray) -> np.ndarray:
        """
        X       : (T, D) multivariate signal
        returns : (D, L) coefficient matrix
        """
        return np.stack([self.dwt(X[:, d]) for d in range(X.shape[1])])

    def idwt2d(self, C: np.ndarray, original_length: int) -> np.ndarray:
        """
        C               : (D, L) coefficient matrix
        original_length : target T
        returns         : (T, D) reconstructed signal
        """
        return np.stack(
            [self.idwt(C[d], original_length) for d in range(C.shape[0])],
            axis=1,
        )


if __name__ == "__main__":
    haar = HaarWT(levels=3)
    rng  = np.random.default_rng(0)

    x     = rng.standard_normal(14).astype(np.float32)
    c     = haar.dwt(x)
    x_hat = haar.idwt(c, original_length=14)
    print(f"1-D round-trip error (T=14): {np.abs(x - x_hat).max():.2e}")

    X     = rng.standard_normal((14, 12)).astype(np.float32)
    C     = haar.dwt2d(X)
    X_hat = haar.idwt2d(C, original_length=14)
    print(f"2-D round-trip error (T=14, D=12): {np.abs(X - X_hat).max():.2e}")
    print(f"Coefficient shape: {C.shape}   (D=12, L={haar.coeff_len(14)})")
