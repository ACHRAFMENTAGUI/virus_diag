import numpy as np
import pytest
from core.logistic_regression import LogisticRegression
from utils.preprocessing import StandardScaler
from utils.metrics import Metrics
from app.interface_clinique import ClinicalPredictor


def test_metrics():
  y_true = np.array([0, 1, 1, 0])
  y_pred = np.array([0, 1, 0, 0])
  acc = Metrics.accuracy(y_true, y_pred)
  assert acc == 0.75


def test_clinical_predictor():
  # Modèle factice (mock) implémentant predict
  class DummyModel:
    def predict(self, X):
      return np.array([0.8])

  scaler = StandardScaler()
  scaler.mean = np.zeros(2)
  scaler.scale = np.ones(2)

  model = DummyModel()
  predictor = ClinicalPredictor(model, scaler)

  result = predictor.diagnose([38.5, 90])
  assert result == "Infecté"