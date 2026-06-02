"""
data_pipeline.py — CMAPSS data loading, windowing, and normalisation.

Expected file layout (from config.DATA_DIR):
    train_FD001.txt   26 space-separated columns, no header
    test_FD001.txt    same format
    RUL_FD001.txt     one integer per line (100 values for FD001)
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from config import Config, cfg


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CMAPSSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)   # (N, T, D)
        self.y = torch.tensor(y, dtype=torch.long)       # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# File parsers
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw(path: Path) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            v = line.strip().split()
            if v:
                rows.append([float(x) for x in v])
    return np.array(rows, dtype=np.float32)


def _build_engines(arr: np.ndarray, rul_arr: np.ndarray | None = None,
                   c: Config = cfg) -> list:
    """
    Returns list of (sensors, labels, rul, engine_id) tuples.
    sensors: (n_cycles, D)  float32
    labels:  (n_cycles,)    int
    rul:     (n_cycles,)    int
    """
    engines = []
    for eid in np.unique(arr[:, 0]).astype(int):
        rows   = arr[arr[:, 0] == eid]
        cycles = rows[:, 1]
        sensors = rows[:, 5:][:, c.KEEP_IDX].astype(np.float32)   # (n, D)

        if rul_arr is None:
            rul = np.minimum(int(cycles.max()) - cycles, c.RUL_CAP).astype(int)
        else:
            last_rul = int(rul_arr[eid - 1])
            rul = np.minimum(
                int(cycles.max()) - cycles + last_rul, c.RUL_CAP
            ).astype(int)

        labels = np.where(
            rul > c.HEALTHY_THRESH, 0,
            np.where(rul > c.CRITICAL_THRESH, 1, 2)
        )
        engines.append((sensors, labels, rul, eid))
    return engines


def _build_windows(engines: list, T: int, win_per_eng: int):
    """Slide T-length windows over each engine lifecycle."""
    Xw, yw = [], []
    for sensors, labels, rul, _ in engines:
        n = len(sensors) - T + 1
        if n <= 0:
            continue
        idxs = np.linspace(0, n - 1, min(win_per_eng, n)).astype(int)
        for i in idxs:
            Xw.append(sensors[i : i + T])
            yw.append(int(labels[i + T // 2]))   # label at window centre
    return np.array(Xw, np.float32), np.array(yw, np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_cmapss(c: Config = cfg):
    """
    Load, window, and normalise CMAPSS data.

    Returns
    -------
    X_tr, y_tr : np.ndarray  (N_tr, T, D), (N_tr,)
    X_te, y_te : np.ndarray  (N_te, T, D), (N_te,)
    stats      : dict with 'mu' and 'std' used for normalisation
    """
    subset = c.SUBSET
    d = c.DATA_DIR

    tr_raw = _load_raw(d / f"train_{subset}.txt")
    te_raw = _load_raw(d / f"test_{subset}.txt")
    rul_te = np.loadtxt(d / f"RUL_{subset}.txt")

    train_eng = _build_engines(tr_raw,         c=c)
    test_eng  = _build_engines(te_raw, rul_te, c=c)

    X_tr, y_tr = _build_windows(train_eng, c.T, c.WIN_PER_ENG_TRAIN)
    X_te, y_te = _build_windows(test_eng,  c.T, c.WIN_PER_ENG_TEST)

    # Z-score: fit on train only (no leakage)
    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std( axis=(0, 1), keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std

    print(f"[Data] {subset}  |  Train {X_tr.shape}  |  Test {X_te.shape}")
    print(f"[Data] Class dist (train): {np.bincount(y_tr.astype(int))}")
    print(f"[Data] Class dist (test):  {np.bincount(y_te.astype(int))}")

    return X_tr, y_tr, X_te, y_te, {"mu": mu, "std": std}


def get_dataloaders(X_tr, y_tr, X_te, y_te, c: Config = cfg):
    """Return (train_loader, test_loader) as PyTorch DataLoaders."""
    train_dl = DataLoader(
        CMAPSSDataset(X_tr, y_tr),
        batch_size=c.BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False,
    )
    test_dl = DataLoader(
        CMAPSSDataset(X_te, y_te),
        batch_size=c.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
    )
    return train_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, stats = load_cmapss()
    print(f"mu shape : {stats['mu'].shape}")
    print(f"std shape: {stats['std'].shape}")
    print(f"Sample window stats — min={X_tr[0].min():.3f}, max={X_tr[0].max():.3f}")
