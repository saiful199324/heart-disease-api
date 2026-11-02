import json
from pathlib import Path

import joblib
from fastapi import FastAPI

from app.schemas import HeartFeatures, InfoResponse, PredictionResponse

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

model = joblib.load(MODEL_DIR / "heart_model.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES: list[str] = metadata["features"]

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the presence of heart disease from 13 clinical features.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API. See /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(
        model_type=metadata["model_type"],
        features=FEATURES,
        test_accuracy=metadata["test_accuracy"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartFeatures):
    row = [[getattr(features, name) for name in FEATURES]]
    probability = float(model.predict_proba(row)[0][1])
    return PredictionResponse(heart_disease=probability >= 0.5, probability=round(probability, 4))
