from typing import Tuple
import numpy as np
import pandas as pd

class SequenceBuilder:
    def __init__(self, sequence_length: int, forecast_horizon: int):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features[i : i + self.sequence_length])
            y.append(target[i + self.sequence_length + self.forecast_horizon - 1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def build_sequences(self, df: pd.DataFrame, feature_cols: list, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        features = df[feature_cols].values
        target = df[target_col].values
        return self.create_sequences(features, target)