"""
config.py — Central configuration for the TAGFC-AQI pipeline.
Applied to AQI India daily dataset (city_day.csv).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

HERE     = Path(__file__).parent
DATA_DIR = HERE.parent / "data" / "raw" / "AQI_INDIA"


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR:   Path = DATA_DIR
    OUTPUT_DIR: Path = HERE / "outputs"
    MODEL_DIR:  Path = HERE / "outputs" / "models"
    FIG_DIR:    Path = HERE / "outputs" / "figures"

    # ── Dataset ────────────────────────────────────────────────────────────────
    DATA_FILE: str = "city_day.csv"
    T:  int = 14          # window length: 14 days
    D:  int = 12          # number of pollutant features
    K:  int = 3           # classes: 0=Good  1=Moderate  2=Poor

    # AQI_Bucket → class mapping
    # Good, Satisfactory        → 0 (Good)
    # Moderate                  → 1 (Moderate)
    # Poor, Very Poor, Severe   → 2 (Poor)
    BUCKET_MAP: dict = field(default_factory=lambda: {
        "Good":        0,
        "Satisfactory": 0,
        "Moderate":    1,
        "Poor":        2,
        "Very Poor":   2,
        "Severe":      2,
    })
    CLASS_NAMES: dict = field(default_factory=lambda: {
        0: "Good",
        1: "Moderate",
        2: "Poor",
    })

    SENSOR_NAMES: tuple = (
        "PM2.5", "PM10", "NO", "NO2", "NOx",
        "NH3", "CO", "SO2", "O3",
        "Benzene", "Toluene", "Xylene",
    )

    MIN_CITY_DAYS: int = 100   # drop cities with fewer than this many valid rows

    # ── Train / Test split ─────────────────────────────────────────────────────
    TEST_FRAC: float = 0.20
    VAL_FRAC:  float = 0.10

    # ── Transformer ────────────────────────────────────────────────────────────
    D_MODEL:  int   = 64
    N_HEADS:  int   = 4
    D_FF:     int   = 128
    N_LAYERS: int   = 3
    DROPOUT:  float = 0.1

    TRAIN_LR:   float = 1e-3
    TRAIN_WD:   float = 1e-4
    EPOCHS:     int   = 150
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
    N_EXPLAIN:   int   = 60     # increase to 150+ for publication
    ALPHA:       float = 0.05
    RANDOM_SEED: int   = 42

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.FIG_DIR.mkdir(parents=True, exist_ok=True)


cfg = Config()
