"""
config.py — Central configuration for the TAGFC pipeline.
All hyperparameters and paths live here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
CMAPSS_DIR = HERE.parent / "data" / "raw" / "CMAPSS"


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR:   Path = CMAPSS_DIR
    OUTPUT_DIR: Path = HERE / "outputs"
    MODEL_DIR:  Path = HERE / "outputs" / "models"
    FIG_DIR:    Path = HERE / "outputs" / "figures"

    # ── Dataset ────────────────────────────────────────────────────────────────
    SUBSET: str = "FD001"       # FD001 | FD002 | FD003 | FD004
    T:      int = 30            # window length (timesteps)
    D:      int = 14            # sensors kept after removing constants
    K:      int = 3             # classes: 0=Healthy 1=Degrading 2=Critical
    RUL_CAP: int = 125          # cap applied to RUL values

    HEALTHY_THRESH:  int = 120  # RUL > 120  → class 0
    CRITICAL_THRESH: int = 30   # RUL <= 30  → class 2;  else class 1

    WIN_PER_ENG_TRAIN: int = 6
    WIN_PER_ENG_TEST:  int = 4

    # 0-based indices within the 21-sensor block (columns 5-25 of raw file)
    CONST_IDX: tuple = (0, 4, 5, 9, 15, 17, 18)
    KEEP_IDX:  tuple = (1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20)
    SENSOR_NAMES: tuple = (
        's2', 's3', 's4', 's7', 's8', 's9',
        's11', 's12', 's13', 's14', 's15',
        's17', 's20', 's21',
    )

    # ── Transformer ────────────────────────────────────────────────────────────
    D_MODEL:  int   = 64
    N_HEADS:  int   = 4
    D_FF:     int   = 128
    N_LAYERS: int   = 3
    DROPOUT:  float = 0.1

    TRAIN_LR: float = 1e-3
    TRAIN_WD: float = 1e-4
    EPOCHS:   int   = 200
    BATCH_SIZE: int = 64

    # ── Haar Wavelet ───────────────────────────────────────────────────────────
    WAV_LEVELS: int = 3

    # ── TAGFC Optimiser ────────────────────────────────────────────────────────
    LAMBDA_SPARSE:   float = 0.02
    LAMBDA_CROSS:    float = 0.01
    LAMBDA_MANIFOLD: float = 0.10
    LR_OPT:          float = 0.05
    MAX_ITER:        int   = 300
    PATIENCE:        int   = 25
    DELTA_BOUND:     float = 2.5
    GRAD_SMOOTH_SIGMA: float = 1.5

    # ── Evaluation ─────────────────────────────────────────────────────────────
    N_EXPLAIN:   int   = 60     # increase to 200 for publication
    ALPHA:       float = 0.05   # family-wise error rate (Bonferroni applied)
    RANDOM_SEED: int   = 42

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.FIG_DIR.mkdir(parents=True, exist_ok=True)
        self.KEEP_IDX = list(self.KEEP_IDX)


# Singleton used throughout the project
cfg = Config()
