"""
Implémentation d'une Régression Logistique sur mesure héritant de Model.
"""

import numpy as np
from core.model import Model
from core.optimizer import Optimizer, SGD


class LogisticRegression(Model):
  """Modèle de classification par régression logistique binaire."""

  def __init__(self, optimizer: Optimizer | None = None, epochs: int = 1000):
    self.optimizer = optimizer or SGD(lr=0.01)
    self.epochs = epochs
    self.weights: np.ndarray | None = None
    self.bias: float = 0.0

  def _sigmoid(self, z: np.ndarray) -> np.ndarray:
    """Fonction sigmoïde numériquement stable."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

  def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Entraîne le modèle par descente de gradient."""
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0.0

    for _ in range(self.epochs):
      linear_model = np.dot(X, self.weights) + self.bias
      y_pred = self._sigmoid(linear_model)

      # Calcul des gradients
      dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
      db = (1 / n_samples) * np.sum(y_pred - y)

      # Mise à jour via l'optimiseur
      self.weights, self.bias = self.optimizer.update(
          self.weights, dw, self.bias, db
      )

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Retourne les probabilités prédites."""
    if self.weights is None:
      raise ValueError("Le modèle n'est pas entraîné.")
    linear_model = np.dot(X, self.weights) + self.bias
    return self._sigmoid(linear_model)