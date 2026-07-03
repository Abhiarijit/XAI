"""
data_pipeline.py — NASA C-MAPSS FD001 data loading, windowing, normalisation.

Pipeline
--------
1. Load train_FD001.txt → build RUL labels (capped at 125)
2. Discretise RUL → 3 classes: Healthy(0) Degrading(1) Critical(2)
3. Remove 7 constant sensors
4. Slide T=30 windows, stride=1
5. Z-score normalise (fit on train only)
6. Load test_FD001.txt + RUL_FD001.txt for test set
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import Config, cfg


# All 21 raw sensor column names (in file order after id/cycle/op columns)
_ALL_SENSORS = [f"s{i}" for i in range(1, 22)]

# Column indices in the raw file: id(0) cycle(1) op1(2) op2(3) op3(4) s1..s21(5..25)
_RAW_COLS = ["id", "cycle", "op1", "op2", "op3"] + _ALL_SENSORS


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CMAPSSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rul_to_class(rul: int, c: Config) -> int:
    if rul > c.HEALTHY_THRESH:
        return 0   # Healthy
    elif rul > c.CRITICAL_THRESH:
        return 1   # Degrading
    else:
        return 2   # Critical


def _load_txt(path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def _keep_sensor_cols(c: Config) -> list[int]:
    """Column indices (0-based within sensor block) to keep."""
    keep = [i for i, s in enumerate(_ALL_SENSORS) if s not in c.DROP_SENSORS]
    return keep   # 14 indices


# ─────────────────────────────────────────────────────────────────────────────
# Train set: build windows from full engine trajectories
# ─────────────────────────────────────────────────────────────────────────────

def _build_train(c: Config):
    raw = _load_txt(c.DATA_DIR / c.TRAIN_FILE)   # (N_rows, 26)
    keep = _keep_sensor_cols(c)

    engine_ids = raw[:, 0].astype(int)
    sensors    = raw[:, 5:][:, keep]              # (N_rows, 14)

    X_all, y_all = [], []
    for eng_id in np.unique(engine_ids):
        mask   = engine_ids == eng_id
        eng    = sensors[mask]                    # (L_eng, 14)
        L      = len(eng)
        # RUL for each cycle: counts down from (L-1) to 0, capped at RUL_CAP
        ruls   = np.minimum(np.arange(L - 1, -1, -1), c.RUL_CAP)
        labels = np.array([_rul_to_class(r, c) for r in ruls], dtype=np.int64)

        # Slide window of length T
        for i in range(L - c.T + 1):
            X_all.append(eng[i : i + c.T])
            y_all.append(labels[i + c.T - 1])    # label at window end

    return np.array(X_all, np.float32), np.array(y_all, np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Test set: one window per engine (last T cycles)
# ─────────────────────────────────────────────────────────────────────────────

def _build_test(c: Config):
    raw      = _load_txt(c.DATA_DIR / c.TEST_FILE)
    rul_test = _load_txt(c.DATA_DIR / c.RUL_FILE).flatten().astype(int)
    keep     = _keep_sensor_cols(c)

    engine_ids = raw[:, 0].astype(int)
    sensors    = raw[:, 5:][:, keep]

    X_all, y_all = [], []
    for i, eng_id in enumerate(np.unique(engine_ids)):
        mask = engine_ids == eng_id
        eng  = sensors[mask]
        # Use last T cycles; if engine shorter than T, zero-pad at front
        if len(eng) >= c.T:
            window = eng[-c.T:]
        else:
            pad    = np.zeros((c.T - len(eng), c.D), dtype=np.float32)
            window = np.vstack([pad, eng])
        rul   = min(int(rul_test[i]), c.RUL_CAP)
        label = _rul_to_class(rul, c)
        X_all.append(window)
        y_all.append(label)

    return np.array(X_all, np.float32), np.array(y_all, np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_cmapss(c: Config = cfg):
    """
    Returns
    -------
    X_tr, y_tr  : (N_tr, T, D), (N_tr,)
    X_te, y_te  : (N_te, T, D), (N_te,)
    stats       : dict with 'mu', 'std'
    """
    X_tr, y_tr = _build_train(c)
    X_te, y_te = _build_test(c)

    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std( axis=(0, 1), keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std

    print(f"[Data] Train windows : {len(X_tr)}  | classes {np.bincount(y_tr)}")
    print(f"[Data] Test engines  : {len(X_te)}  | classes {np.bincount(y_te)}")

    return X_tr, y_tr, X_te, y_te, {"mu": mu, "std": std}


def get_dataloaders(X_tr, y_tr, X_te, y_te, c: Config = cfg):
    """Weighted random sampler to handle class imbalance in training."""
    counts  = np.bincount(y_tr, minlength=c.K).astype(float)
    weights = 1.0 / np.maximum(counts, 1)
    sample_w = torch.tensor(weights[y_tr], dtype=torch.float)
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_dl = DataLoader(
        CMAPSSDataset(X_tr, y_tr),
        batch_size=c.BATCH_SIZE, sampler=sampler, num_workers=0,
    )
    test_dl = DataLoader(
        CMAPSSDataset(X_te, y_te),
        batch_size=c.BATCH_SIZE, shuffle=False, num_workers=0,
    )
    return train_dl, test_dl
