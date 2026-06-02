"""
data_pipeline.py — NATOPS data loading and normalisation for CoMTE pipeline.

Source files: data/raw/NATOPS/NATOPS_TRAIN.arff
              data/raw/NATOPS/NATOPS_TEST.arff

NATOPS .arff format (UEA multivariate):
    Each row = one time series in a quoted string.
    D=24 channels separated by literal '\\n' within the quoted string.
    Each channel has T=51 comma-separated float values.
    Last token after closing quote comma = class label (float "1.0".."6.0").

Preprocessing:
    1. Parse .arff → (N, T, D) array
    2. Map class labels "1".."6" → 0..5
    3. Z-score normalise (fit on train only)
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from config import Config, cfg


class NATOPSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _parse_arff(path: Path, T: int, D: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse UEA-format NATOPS .arff file.

    Each data row:
        '<ch0_t0,...,ch0_tT-1>\\n<ch1_t0,...>\\n...\\n<ch23_t0,...>',label
    """
    X_list, y_list = [], []
    in_data = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.upper() == "@DATA":
                in_data = True
                continue
            if not in_data:
                continue

            last_comma = line.rfind(",")
            label_str  = line[last_comma + 1:].strip().strip("'\"")
            data_str   = line[:last_comma].strip().strip("'\"")

            channel_strs = data_str.split("\\n")
            if len(channel_strs) != D:
                # Fallback: flat channel-major D*T values
                values = [float(v.strip().strip("'\"")) for v in data_str.split(",")]
                arr = np.array(values, dtype=np.float32).reshape(D, T)
            else:
                arr = np.zeros((D, T), dtype=np.float32)
                for d, ch_str in enumerate(channel_strs):
                    vals = [float(v.strip()) for v in ch_str.split(",")]
                    arr[d, :len(vals)] = vals[:T]

            X_list.append(arr.T)                          # (D, T) → (T, D)
            y_list.append(int(float(label_str)) - 1)      # "1".."6" → 0..5

    X = np.stack(X_list).astype(np.float32)               # (N, T, D)
    y = np.array(y_list, dtype=np.int64)                  # (N,)
    return X, y


def load_natops(c: Config = cfg):
    """
    Load, parse, and normalise NATOPS train/test splits.

    Supports two loading paths:
      - .npy arrays (fast path, created by download script)
      - .arff files (UEA format, parsed manually)

    Returns
    -------
    X_tr, y_tr : (N_tr, T, D), (N_tr,)
    X_te, y_te : (N_te, T, D), (N_te,)
    stats      : dict with 'mu', 'std'
    """
    npy_train = c.DATA_DIR / "X_train.npy"
    npy_test  = c.DATA_DIR / "X_test.npy"

    if npy_train.exists() and npy_test.exists():
        print("[Data] Loading NATOPS from .npy arrays (fast path) ...")
        X_tr = np.load(npy_train).astype(np.float32)
        y_tr = np.load(c.DATA_DIR / "y_train.npy").astype(np.int64)
        X_te = np.load(npy_test).astype(np.float32)
        y_te = np.load(c.DATA_DIR / "y_test.npy").astype(np.int64)
    else:
        train_path = c.DATA_DIR / c.TRAIN_FILE
        test_path  = c.DATA_DIR / c.TEST_FILE

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"NATOPS files not found in {c.DATA_DIR}.\n"
                f"Download NATOPS_TRAIN.arff and NATOPS_TEST.arff from:\n"
                f"  https://timeseriesclassification.com/dataset.php?train=NATOPS\n"
                f"and place them in {c.DATA_DIR}"
            )

        print("[Data] Parsing NATOPS .arff files ...")
        X_tr, y_tr = _parse_arff(train_path, c.T, c.D)
        X_te, y_te = _parse_arff(test_path,  c.T, c.D)

    # Z-score: fit on train only
    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std( axis=(0, 1), keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std

    print(f"[Data] NATOPS  |  Train {X_tr.shape}  |  Test {X_te.shape}")
    print(f"[Data] Class dist (train): {np.bincount(y_tr.astype(int))}")
    print(f"[Data] Class dist (test):  {np.bincount(y_te.astype(int))}")

    return X_tr, y_tr, X_te, y_te, {"mu": mu, "std": std}


def get_dataloaders(X_tr, y_tr, X_te, y_te, c: Config = cfg):
    train_dl = DataLoader(
        NATOPSDataset(X_tr, y_tr),
        batch_size=c.BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False,
    )
    test_dl = DataLoader(
        NATOPSDataset(X_te, y_te),
        batch_size=c.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False,
    )
    return train_dl, test_dl


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, stats = load_natops()
    print(f"Sample shape : {X_tr[0].shape}")
    print(f"Min/Max (train): {X_tr.min():.3f} / {X_tr.max():.3f}")
