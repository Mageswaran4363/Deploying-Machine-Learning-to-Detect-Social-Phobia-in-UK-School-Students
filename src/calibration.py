from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibratedModel:
    """Wraps a fitted sklearn classifier with a Platt sigmoid calibration layer."""

    def __init__(self, base_model) -> None:
        self.base_model  = base_model
        self._calibrator = LogisticRegression(C=1.0, random_state=42)
        self._fitted     = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlattCalibratedModel":
        scores = self.base_model.decision_function(X).reshape(-1, 1)
        self._calibrator.fit(scores, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.base_model.decision_function(X).reshape(-1, 1)
        return self._calibrator.predict_proba(scores)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
