"""
Implémentation d'un Perceptron Multicouche (Réseau de Neurones simple) à 1 couche cachée.
"""

import numpy as np
from core.model import Model


class NeuralNetwork(Model):
  """Réseau de neurones simple à une couche cachée pour classification binaire."""

  def __init__(self, hidden_dim: int = 16, lr: float = 0.01, epochs: int = 1000):
    self.hidden_dim = hidden_dim
    self.lr = lr
    self.epochs = epochs
    self.W1: np.ndarray | None = None
    self.b1: np.ndarray | None = None
    self.W2: np.ndarray | None = None
    self.b2: float | None = None

  def _relu(self, z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

  def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

  def _sigmoid(self, z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

  def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    n_samples, n_features = X.shape
    np.random.seed(42)

    self.W1 = np.random.randn(n_features, self.hidden_dim) * np.sqrt(2.0 / n_features)
    self.b1 = np.zeros((1, self.hidden_dim))
    self.W2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
    self.b2 = 0.0

    y_reshaped = y.reshape(-1, 1)

    for _ in range(self.epochs):
      # Forward pass
      z1 = np.dot(X, self.W1) + self.b1
      a1 = self._relu(z1)
      z2 = np.dot(a1, self.W2) + self.b2
      y_pred = self._sigmoid(z2)

      # Backward pass
      dz2 = y_pred - y_reshaped
      dW2 = np.dot(a1.T, dz2) / n_samples
      db2 = np.sum(dz2) / n_samples

      dz1 = np.dot(dz2, self.W2.T) * self.relu_deriv = self._relu_derivative(z1)
      dW1 = np.dot(X.T, dz1) / n_samples
      db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples

      # Gradient descent update
      self.W1 -= self.lr * dW1
      self.b1 -= self.lr * db1
      self.W2 -= self.lr * dW2
      self.b2 -= self.lr * db2

  def predict(self, X: np.ndarray) -> np.ndarray:
    if self.W1 is None:
      raise ValueError("Le réseau de neurones n'est pas entraîné.")
    z1 = np.dot(X, self.W1) + self.b1
    a1 = self._relu(z1)
    z2 = np.dot(a1, self.W2) + self.b2
    return self._sigmoid(z2).flatten()