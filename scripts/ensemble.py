"""Kept in its own module so a saved model can be loaded by any script."""
import numpy as np


class SeedEnsemble:
    """Average of identical models trained with different random seeds (less luck)."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)
