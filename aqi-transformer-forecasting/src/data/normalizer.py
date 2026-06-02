class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.feature_min = None
        self.feature_max = None
        self.target_min = None
        self.target_max = None
        self._is_fitted = False

    def fit(self, features, target):
        self.feature_min = features.min(axis=0)
        self.feature_max = features.max(axis=0)
        self.target_min = target.min()
        self.target_max = target.max()
        self._is_fitted = True

    def transform_features(self, features):
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call 'fit' before 'transform_features'.")
        return (features - self.feature_min) / (self.feature_max - self.feature_min) * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def transform_target(self, target):
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call 'fit' before 'transform_target'.")
        return (target - self.target_min) / (self.target_max - self.target_min) * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def inverse_transform_target(self, target):
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call 'fit' before 'inverse_transform_target'.")
        return (target - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * (self.target_max - self.target_min) + self.target_min

    def get_params(self):
        return {
            'feature_min': self.feature_min,
            'feature_max': self.feature_max,
            'target_min': self.target_min,
            'target_max': self.target_max,
        }