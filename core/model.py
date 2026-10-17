"""
Définition de la classe abstraite Model (Interface / ABC).
"""

from abc import ABC, abstractmethod
import pickle
from pathlib import Path
import numpy as np


class Model(ABC):
  """Classe abstraite de base pour tous les modèles de Machine Learning."""

  @abstractmethod
  def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Entraîne le modèle sur les données X et y."""
    pass

  @abstractmethod
  def predict(self, X: np.ndarray) -> np.ndarray:
    """Réalise des prédictions (probabilités ou classes) pour X."""
    pass

  def save(self, filepath: str | Path) -> None:
    """Sauvegarde le modèle sur le disque via pickle."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
      pickle.dump(self, f)

  @classmethod
  def load(cls, filepath: str | Path) -> "Model":
    """Charge un modèle sérialisé depuis le disque."""
    path = Path(filepath)
    if not path.exists():
      raise FileNotFoundError(f"Fichier modèle introuvable : {path}")
    with open(path, "rb") as f:
      return pickle.load(f)