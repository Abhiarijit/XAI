"""
config.py — Central configuration for the TAGFC-CMAPSS pipeline.
Applied to NASA C-MAPSS FD001 turbofan degradation dataset.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

HERE     = Path(__file__).parent
DATA_DIR = HERE.parent / "data" / "raw" / "CMAPSS"


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR:   Path = DATA_DIR
    OUTPUT_DIR: Path = HERE / "outputs"
    MODEL_DIR:  Path = HERE / "outputs" / "models"
    FIG_DIR:    Path = HERE / "outputs" / "figures"

    # ── Dataset ────────────────────────────────────────────────────────────────
    TRAIN_FILE: str = "train_FD001.txt"
    TEST_FILE:  str = "test_FD001.txt"
    RUL_FILE:   str = "RUL_FD001.txt"

    T:  int = 30    # sliding window length (cycles)
    D:  int = 14    # informative sensors after removing 7 constant ones
    K:  int = 3     # classes: 0=Healthy  1=Degrading  2=Critical

    RUL_CAP:          int = 125   # piecewise-linear RUL cap
    HEALTHY_THRESH:   int = 120   # RUL > 120  → Healthy
    CRITICAL_THRESH:  int = 30    # RUL ≤ 30   → Critical

    # Sensors to REMOVE (constant across all cycles in FD001)
    DROP_SENSORS: tuple = ("s1", "s5", "s6", "s10", "s16", "s18", "s19")

    CLASS_NAMES: dict = field(default_factory=lambda: {
        0: "Healthy",
        1: "Degrading",
        2: "Critical",
    })

    SENSOR_NAMES: tuple = (
        "s2", "s3", "s4", "s7", "s8", "s9",
        "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
    )

    # ── Train / Val split ──────────────────────────────────────────────────────
    VAL_FRAC:    float = 0.20
    RANDOM_SEED: int   = 42

    # ── Transformer ────────────────────────────────────────────────────────────
    D_MODEL:  int   = 64
    N_HEADS:  int   = 4
    D_FF:     int   = 128
    N_LAYERS: int   = 3
    DROPOUT:  float = 0.1

    TRAIN_LR:   float = 1e-3
    TRAIN_WD:   float = 1e-4
    EPOCHS:     int   = 200
    BATCH_SIZE: int   = 64

    # ── Haar Wavelet ───────────────────────────────────────────────────────────
    WAV_LEVELS: int = 3

    # ── TAGFC Optimiser ────────────────────────────────────────────────────────
    LAMBDA_SPARSE:     float = 0.02
    LAMBDA_CROSS:      float = 0.01
    LAMBDA_MANIFOLD:   float = 0.10
    LR_OPT:            float = 0.05
    MAX_ITER:          int   = 300
    PATIENCE:          int   = 25
    DELTA_BOUND:       float = 2.5
    GRAD_SMOOTH_SIGMA: float = 1.5

    # ── Evaluation ─────────────────────────────────────────────────────────────
    N_EXPLAIN:   int   = 60
    ALPHA:       float = 0.05

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.FIG_DIR.mkdir(parents=True, exist_ok=True)


cfg = Config()
