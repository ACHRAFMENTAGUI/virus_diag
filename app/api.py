"""
API FastAPI pour exposer le point de terminaison de diagnostic /predict.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from pathlib import Path

from core.logistic_regression import LogisticRegression
from utils.preprocessing import StandardScaler
from app.interface_clinique import ClinicalPredictor
from core.dataset import Dataset

app = FastAPI(title="VirusDiag API", version="1.0")

# Initialisation globale pour l'API
predictor: ClinicalPredictor | None = None


@app.on_event("startup")
def startup_event():
  global predictor
  data_path = Path("data/patient_data.csv")
  if not data_path.exists():
    return
  dataset = Dataset(data_path)
  X, y = dataset.prepare_data()
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  model = LogisticRegression(epochs=500)
  model.fit(X_scaled, y)
  predictor = ClinicalPredictor(model, scaler)


class PatientPayload(BaseModel):
  age: int
  temperature: float
  oxygen_saturation: float
  heart_rate: int
  white_blood_cells: int
  cough: int
  fatigue: int
  loss_of_taste: int
  contact_history: int


@app.post("/predict")
def predict_endpoint(payload: PatientPayload):
  if predictor is None:
    raise HTTPException(status_code=500, detail="Modèle non initialisé.")

  features = [
      payload.age,
      payload.temperature,
      payload.oxygen_saturation,
      payload.heart_rate,
      payload.white_blood_cells,
      payload.cough,
      payload.fatigue,
      payload.loss_of_taste,
      payload.contact_history,
  ]
  diagnosis = predictor.diagnose(features)
  return {"diagnostic": diagnosis}