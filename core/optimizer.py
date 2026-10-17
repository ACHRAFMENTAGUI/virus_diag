"""
Classes d'optimisation pour la descente de gradient.
"""

from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
  """Classe abstraite pour les optimiseurs."""

  @abstractmethod
  def update(
      self, weights: np.ndarray, grad_weights: np.ndarray, bias: float, grad_bias: float
  ) -> tuple[np.ndarray, float]:
    """Met à jour les poids et biais et retourne les nouvelles valeurs."""
    pass


class SGD(Optimizer):
  """Optimiseur Stochastique par Descente de Gradient (SGD)."""

  def __init__(self, lr: float = 0.01):
    self.lr = lr

  def update(
      self, weights: np.ndarray, grad_weights: np.ndarray, bias: float, grad_bias: float
  ) -> tuple[np.ndarray, float]:
    new_weights = weights - self.lr * grad_weights
    new_bias = bias - self.lr * grad_bias
    return new_weights, new_bias


class Adam(Optimizer):
  """Optimiseur Adam simplifié."""

  def __init__(
      self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8
  ):
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.m_w = 0
    self.v_w = 0
    self.m_b = 0
    self.v_b = 0
    self.t = 0

  def update(
      self, weights: np.ndarray, grad_weights: np.ndarray, bias: float, grad_bias: float
  ) -> tuple[np.ndarray, float]:
    self.t += 1
    if isinstance(self.m_w, int):
      self.m_w = np.zeros_like(weights)
      self.v_w = np.zeros_like(weights)
      self.m_b = 0.0
      self.v_b = 0.0

    self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_weights
    self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_weights ** 2)
    m_w_corr = self.m_w / (1 - self.beta1 ** self.t)
    v_w_corr = self.v_w / (1 - self.beta2 ** self.t)
    new_weights = weights - self.lr * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)

    self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_bias
    self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_bias ** 2)
    m_b_corr = self.m_b / (1 - self.beta1 ** self.t)
    v_b_corr = self.v_b / (1 - self.beta2 ** self.t)
    new_bias = bias - self.lr * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)

    return new_weights, new_bias