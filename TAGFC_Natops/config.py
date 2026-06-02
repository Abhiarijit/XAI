"""
config.py — Central configuration for the TAGFC-NATOPS pipeline.

Dataset: NATOPS (Naval Air Training and Operating Procedures Standardization)
Task   : 6-class gesture recognition from 24-channel body-worn IMU sensors
Shape  : (N, 51, 24)  →  T=51 timesteps, D=24 channels, K=6 classes
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

HERE     = Path(__file__).parent
DATA_DIR = HERE.parent / "data" / "raw" / "NATOPS"


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR:   Path = DATA_DIR
    OUTPUT_DIR: Path = HERE / "outputs"
    MODEL_DIR:  Path = HERE / "outputs" / "models"
    FIG_DIR:    Path = HERE / "outputs" / "figures"

    # ── Dataset ────────────────────────────────────────────────────────────────
    TRAIN_FILE: str = "NATOPS_TRAIN.arff"
    TEST_FILE:  str = "NATOPS_TEST.arff"

    T:  int = 51    # timesteps per series
    D:  int = 24    # sensor channels
    K:  int = 6     # gesture classes (1-indexed in .arff → mapped to 0..5)

    CLASS_NAMES: dict = field(default_factory=lambda: {
        0: "I have command",
        1: "All clear",
        2: "Not clear",
        3: "Spread wings",
        4: "Fold wings",
        5: "Lock wings",
    })

    # 24 sensor channel names (hands/elbows/wrists/thumbs — x,y,z per joint)
    SENSOR_NAMES: tuple = (
        "Hand_R_x", "Hand_R_y", "Hand_R_z",
        "Hand_L_x", "Hand_L_y", "Hand_L_z",
        "Elbow_R_x","Elbow_R_y","Elbow_R_z",
        "Elbow_L_x","Elbow_L_y","Elbow_L_z",
        "Wrist_R_x","Wrist_R_y","Wrist_R_z",
        "Wrist_L_x","Wrist_L_y","Wrist_L_z",
        "Thumb_R_x","Thumb_R_y","Thumb_R_z",
        "Thumb_L_x","Thumb_L_y","Thumb_L_z",
    )

    # ── Transformer ────────────────────────────────────────────────────────────
    D_MODEL:  int   = 64
    N_HEADS:  int   = 4
    D_FF:     int   = 128
    N_LAYERS: int   = 3
    DROPOUT:  float = 0.1

    TRAIN_LR:   float = 1e-3
    TRAIN_WD:   float = 1e-4
    EPOCHS:     int   = 200
    BATCH_SIZE: int   = 32

    # ── Haar Wavelet ───────────────────────────────────────────────────────────
    WAV_LEVELS: int = 3        # T=51 → L=64 coefficients

    # ── TAGFC Optimiser ────────────────────────────────────────────────────────
    LAMBDA_SPARSE:     float = 0.02
    LAMBDA_CROSS:      float = 0.01
    LAMBDA_MANIFOLD:   float = 0.10
    LR_OPT:            float = 0.05
    MAX_ITER:          int   = 300
    PATIENCE:          int   = 25
    DELTA_BOUND:       float = 2.5
    GRAD_SMOOTH_SIGMA: float = 1.5

    # ── Baselines ──────────────────────────────────────────────────────────────
    COMTE_WINDOW_FRAC: float = 0.15
    ABCF_WINDOW_FRAC:  float = 0.15

    # ── Evaluation ─────────────────────────────────────────────────────────────
    N_EXPLAIN:   int   = 30
    ALPHA:       float = 0.05
    RANDOM_SEED: int   = 42

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.FIG_DIR.mkdir(parents=True, exist_ok=True)


cfg = Config()
