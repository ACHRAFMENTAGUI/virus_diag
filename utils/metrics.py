"""
Calcul des métriques d'évaluation de classification.
"""

import numpy as np


class Metrics:
  """Regroupe les métriques d'évaluation clinique."""

  @staticmethod
  def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Retourne la matrice de confusion [[TN, FP], [FN, TP]]."""
    y_t = y_true.astype(int)
    y_p = y_pred.astype(int)
    tn = np.sum((y_t == 0) & (y_p == 0))
    fp = np.sum((y_t == 0) & (y_p == 1))
    fn = np.sum((y_t == 1) & (y_p == 0))
    tp = np.sum((y_t == 1) & (y_p == 1))
    return np.array([[tn, fp], [fn, tp]])

  @staticmethod
  def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = Metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    return float((tp + tn) / total) if total > 0 else 0.0

  @staticmethod
  def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = Metrics.confusion_matrix(y_true, y_pred)
    _, fp, _, tp = cm.ravel()
    denominator = tp + fp
    return float(tp / denominator) if denominator > 0 else 0.0

  @staticmethod
  def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = Metrics.confusion_matrix(y_true, y_pred)
    _, _, fn, tp = cm.ravel()
    denominator = tp + fn
    return float(tp / denominator) if denominator > 0 else 0.0

  @staticmethod
  def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = Metrics.precision(y_true, y_pred)
    r = Metrics.recall(y_true, y_pred)
    denominator = p + r
    return float(2 * (p * r) / denominator) if denominator > 0 else 0.0