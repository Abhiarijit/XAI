from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AQIDataset(Dataset):
    def __init__(self, data, target_col, feature_cols):
        self.data = data
        self.target_col = target_col
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[self.feature_cols].iloc[idx].values.astype(np.float32)
        target = self.data[self.target_col].iloc[idx].astype(np.float32)
        return features, target

    def get_features(self):
        return self.data[self.feature_cols].values

    def get_targets(self):
        return self.data[self.target_col].values