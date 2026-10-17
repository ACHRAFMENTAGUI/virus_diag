"""
Module d'entraînement des modèles de diagnostic.
"""

import numpy as np
from core.model import Model
from utils.preprocessing import StandardScaler


class Trainer:
  """Orchestre le processus d'apprentissage du modèle."""

  def __init__(self, model: Model):
    self.model = model
    self.scaler = StandardScaler()

  def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Model:
    """Normalise les données et entraîne le modèle."""
    X_scaled = self.scaler.fit_transform(X_train)
    self.model.fit(X_scaled, y_train, **kwargs)
    return self.model

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Prédit sur de nouvelles données normalisées."""
    X_scaled = self.scaler.transform(X)
    return self.model.predict(X_scaled)