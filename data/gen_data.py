"""
Script de génération d'un dataset médical synthétique et réaliste pour le diagnostic viral.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_patient_dataset(
    num_samples: int = 800, output_path: str = "data/patient_data.csv"
) -> None:
  """Génère un dataset de patients avec des corrélations réalistes et du bruit."""
  np.random.seed(42)

  patient_id = [f"PAT_{i:04d}" for i in range(1, num_samples + 1)]
  age = np.random.randint(18, 91, size=num_samples)
  temperature = np.round(np.random.normal(37.0, 1.1, size=num_samples), 1)
  temperature = np.clip(temperature, 36.0, 40.5)

  oxygen_saturation = np.round(
      np.random.normal(96.0, 3.5, size=num_samples), 1
  )
  oxygen_saturation = np.clip(oxygen_saturation, 85.0, 100.0)

  heart_rate = np.random.randint(50, 131, size=num_samples)
  white_blood_cells = np.random.randint(3000, 15001, size=num_samples)

  cough = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
  fatigue = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
  loss_of_taste = np.random.choice([0, 1], size=num_samples, p=[0.75, 0.25])
  contact_history = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])

  # Logique réaliste pour la cible 'infected' avec ajout de bruit
  risk_score = (
      (temperature > 38.0).astype(int) * 3
      + (oxygen_saturation < 92.0).astype(int) * 4
      + loss_of_taste * 3
      + cough * 2
      + fatigue * 1
      + contact_history * 2
      + (white_blood_cells < 4000).astype(int) * 1
  )

  # Normalisation du score de risque en probabilité via sigmoïde et ajout de bruit stochastique
  probabilities = 1 / (
      1 + np.exp(-(risk_score - np.median(risk_score)) / 1.5)
  )
  noise = np.random.uniform(0, 1, size=num_samples)
  infected = (probabilities > (0.5 + np.random.normal(0, 0.15, size=num_samples))).astype(
      int
  )
  infected = np.clip(infected, 0, 1)

  df = pd.DataFrame({
      "patient_id": patient_id,
      "age": age,
      "temperature": temperature,
      "oxygen_saturation": oxygen_saturation,
      "heart_rate": heart_rate,
      "white_blood_cells": white_blood_cells,
      "cough": cough,
      "fatigue": fatigue,
      "loss_of_taste": loss_of_taste,
      "contact_history": contact_history,
      "infected": infected,
  })

  output_file = Path(output_path)
  output_file.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(output_file, index=False)
  print(f"[+] Dataset généré avec succès : {output_file} ({num_samples} lignes)")


if __name__ == "__main__":
  generate_patient_dataset()