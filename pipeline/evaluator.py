"""
Module d'évaluation des performances des modèles.
"""

import numpy as np
from utils.metrics import Metrics


class Evaluator:
  """Évalue un modèle sur un ensemble de test."""

  def __init__(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5):
    self.y_true = y_true
    self.y_pred = (y_pred_proba >= threshold).astype(int)

  def get_report(self) -> dict[str, float | np.ndarray]:
    """Génère un dictionnaire complet des métriques."""
    return {
        "accuracy": Metrics.accuracy(self.y_true, self.y_pred),
        "precision": Metrics.precision(self.y_true, self.y_pred),
        "recall": Metrics.recall(self.y_true, self.y_pred),
        "f1_score": Metrics.f1_score(self.y_true, self.y_pred),
        "confusion_matrix": Metrics.confusion_matrix(self.y_true, self.y_pred),
    }