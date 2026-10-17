"""
Module de gestion et de découpage des jeux de données.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


class Dataset:
  """Charge, nettoie et sépare les données médicales pour l'apprentissage."""

  def __init__(self, filepath: str | Path):
    self.filepath = Path(filepath)
    self.data: pd.DataFrame | None = None
    self.X: np.ndarray | None = None
    self.y: np.ndarray | None = None
    self.feature_names: list[str] = []

  def load_data(self) -> pd.DataFrame:
    """Charge le fichier CSV."""
    if not self.filepath.exists():
      raise FileNotFoundError(f"Fichier introuvable : {self.filepath}")
    self.data = pd.read_csv(self.filepath)
    return self.data

  def prepare_data(
      self, target_column: str = "infected"
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Sépare les caractéristiques (features) de la variable cible."""
    if self.data is None:
      self.load_data()

    if target_column not in self.data.columns:
      raise ValueError(f"Colonne cible '{target_column}' absente du dataset.")

    feature_df = self.data.drop(columns=["patient_id", target_column])
    self.feature_names = list(feature_df.columns)
    self.X = feature_df.values.astype(np.float64)
    self.y = self.data[target_column].values.astype(np.float64)
    return self.X, self.y

  def train_test_split(
      self, test_size: float = 0.2, random_state: int = 42
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divise les données en ensembles d'entraînement et de test."""
    if self.X is None or self.y is None:
      self.prepare_data()

    np.random.seed(random_state)
    indices = np.arange(len(self.X))
    np.random.shuffle(indices)

    split_idx = int(len(self.X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = self.X[train_idx], self.X[test_idx]
    y_train, y_test = self.y[train_idx], self.y[test_idx]

    return X_train, X_test, y_train, y_test