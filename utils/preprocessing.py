"""
Module de prétraitement, nettoyage et normalisation des données.
"""

import numpy as np


class StandardScaler:
  """Normaliseur de caractéristiques (moyenne 0, écart-type 1)."""

  def __init__(self):
    self.mean: np.ndarray | None = None
    self.scale: np.ndarray | None = None

  def fit(self, X: np.ndarray) -> "StandardScaler":
    self.mean = np.mean(X, axis=0)
    self.scale = np.std(X, axis=0)
    # Éviter la division par zéro
    self.scale[self.scale == 0] = 1.0
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    if self.mean is None or self.scale is None:
      raise ValueError("Le scaler n'a pas été ajusté (fit).")
    return (X - self.mean) / self.scale

  def fit_transform(self, X: np.ndarray) -> np.ndarray:
    return self.fit(X).transform(X)