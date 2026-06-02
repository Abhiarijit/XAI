"""
data_pipeline.py — AQI India data loading, windowing, and normalisation.

Source file: data/raw/AQI_INDIA/city_day.csv
Columns: City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3,
         Benzene, Toluene, Xylene, AQI, AQI_Bucket

Pipeline
--------
1. Load CSV, parse Date, drop rows with unknown AQI_Bucket
2. Map AQI_Bucket → 3 classes (Good=0, Moderate=1, Poor=2)
3. Forward-fill missing sensor values within each city
4. Drop cities with < MIN_CITY_DAYS valid days
5. Slide T-day windows over each city's time series
6. Stratified train/test split (by city, then by class)
7. Z-score normalise (fit on train only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config, cfg


FEATURE_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene",
]


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AQIDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)   # (N, T, D)
        self.y = torch.tensor(y, dtype=torch.long)       # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw(c: Config) -> pd.DataFrame:
    path = c.DATA_DIR / c.DATA_FILE
    df   = pd.read_csv(path, parse_dates=["Date"])
    df   = df.sort_values(["City", "Date"]).reset_index(drop=True)
    return df


def _map_bucket(df: pd.DataFrame, c: Config) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["AQI_Bucket"].map(c.BUCKET_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def _fill_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill missing sensor values per city."""
    df = df.copy()
    df[FEATURE_COLS] = (
        df.groupby("City")[FEATURE_COLS]
        .transform(lambda s: s.ffill().bfill())
    )
    # Drop rows still missing after filling
    df = df.dropna(subset=FEATURE_COLS)
    return df


def _build_windows(df: pd.DataFrame, T: int, c: Config):
    """
    Slide T-day windows over each city's sorted time series.
    Label = the AQI class at the LAST day of the window.
    """
    Xw, yw = [], []
    for city, grp in df.groupby("City"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        if len(grp) < c.MIN_CITY_DAYS:
            continue
        sensors = grp[FEATURE_COLS].values.astype(np.float32)   # (n_days, D)
        labels  = grp["label"].values.astype(np.int64)           # (n_days,)
        n       = len(grp) - T + 1
        for i in range(n):
            Xw.append(sensors[i : i + T])
            yw.append(int(labels[i + T - 1]))   # label at window end
    return np.array(Xw, np.float32), np.array(yw, np.int64)


def _train_test_split(X: np.ndarray, y: np.ndarray,
                      test_frac: float, seed: int):
    """Stratified split preserving class proportions."""
    rng   = np.random.default_rng(seed)
    idx   = np.arange(len(y))
    tr_idx, te_idx = [], []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n_te    = max(1, int(len(cls_idx) * test_frac))
        te_idx.extend(cls_idx[:n_te].tolist())
        tr_idx.extend(cls_idx[n_te:].tolist())
    tr_idx = np.array(tr_idx)
    te_idx = np.array(te_idx)
    return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_aqi(c: Config = cfg):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_tr, y_tr : (N_tr, T, D), (N_tr,)
    X_te, y_te : (N_te, T, D), (N_te,)
    stats      : dict with 'mu', 'std'
    """
    df = _load_raw(c)
    df = _map_bucket(df, c)
    df = _fill_sensors(df)

    X, y = _build_windows(df, c.T, c)

    X_tr, y_tr, X_te, y_te = _train_test_split(
        X, y, c.TEST_FRAC, c.RANDOM_SEED
    )

    # Z-score: fit on train only
    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std( axis=(0, 1), keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std

    print(f"[Data] Windows  |  Total={len(X)}  Train={len(X_tr)}  Test={len(X_te)}")
    print(f"[Data] Class dist (train): {np.bincount(y_tr.astype(int))}")
    print(f"[Data] Class dist (test):  {np.bincount(y_te.astype(int))}")

    return X_tr, y_tr, X_te, y_te, {"mu": mu, "std": std}


def get_dataloaders(X_tr, y_tr, X_te, y_te, c: Config = cfg):
    """Return (train_loader, test_loader)."""
    train_dl = DataLoader(
        AQIDataset(X_tr, y_tr),
        batch_size=c.BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False,
    )
    test_dl = DataLoader(
        AQIDataset(X_te, y_te),
        batch_size=c.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
    )
    return train_dl, test_dl


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, stats = load_aqi()
    print(f"X_tr shape : {X_tr.shape}   dtype={X_tr.dtype}")
    print(f"Sample min/max: {X_tr[0].min():.3f} / {X_tr[0].max():.3f}")
