"""
Classe ClinicalPredictor et interface utilisateur Streamlit.
"""
import sys
from pathlib import Path

# Ajoute dynamiquement la racine du projet (Virus_diag) au chemin Python
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
  sys.path.append(str(root_dir))

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from core.model import Model
from utils.preprocessing import StandardScaler


class ClinicalPredictor:
  """Encapsule un modèle entraîné et le scaler pour formuler un diagnostic clinique."""

  def __init__(self, model: Model, scaler: StandardScaler):
    self.model = model
    self.scaler = scaler

  def diagnose(self, patient_data: np.ndarray | list[float]) -> str:
    """Reçoit les données d'un patient et retourne 'Infecté' ou 'Sain'."""
    data_array = np.array(patient_data, dtype=np.float64)
    if data_array.ndim == 1:
      data_array = data_array.reshape(1, -1)

    scaled_data = self.scaler.transform(data_array)
    probability = self.model.predict(scaled_data)[0]

    return "Infecté" if probability >= 0.5 else "Sain"


# Interface Streamlit
def main_streamlit():
  st.set_page_config(page_title="VirusDiag - Diagnostic Clinique", page_icon="🩺")
  st.title("🩺 Système d'Aide au Diagnostic Viral (VirusDiag)")
  st.markdown("Veuillez renseigner les paramètres cliniques du patient.")

  # Chargement simulé ou par défaut d'un modèle pré-entraîné pour la démo
  from core.logistic_regression import LogisticRegression
  from core.dataset import Dataset

  @st.cache_resource
  def load_demo_predictor():
    dataset = Dataset(Path("data/patient_data.csv"))
    X, y = dataset.prepare_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(epochs=500)
    model.fit(X_scaled, y)
    return ClinicalPredictor(model, scaler)

  try:
    predictor = load_demo_predictor()
  except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.info("Veuillez d'abord exécuter le script de génération de données et entraîner le modèle.")
    return

  col1, col2 = st.columns(2)

  with col1:
    age = st.slider("Âge", 18, 90, 45)
    temperature = st.number_input("Température (°C)", 36.0, 40.5, 37.2, 0.1)
    oxygen_saturation = st.slider("Saturation en oxygène (%)", 85, 100, 96)
    heart_rate = st.slider("Fréquence cardiaque (bpm)", 50, 130, 75)
    white_blood_cells = st.number_input("Globules blancs (/µL)", 3000, 15000, 6500, 100)

  with col2:
    cough = st.selectbox("Toux", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    fatigue = st.selectbox("Fatigue", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    loss_of_taste = st.selectbox("Perte de goût / odorat", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    contact_history = st.selectbox("Historique de contact", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

  if st.button("Lancer le Diagnostic", type="primary"):
    patient_features = [
        age,
        temperature,
        oxygen_saturation,
        heart_rate,
        white_blood_cells,
        cough,
        fatigue,
        loss_of_taste,
        contact_history,
    ]
    result = predictor.diagnose(patient_features)
    if result == "Infecté":
      st.error(f"Résultat du diagnostic : **{result}** (Risque élevé détecté)")
    else:
      st.success(f"Résultat du diagnostic : **{result}** (Patient stable)")


if __name__ == "__main__":
  main_streamlit()