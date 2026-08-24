"""
main.py — FastAPI backend for the phishing website detector.

Endpoints:
    GET  /health          -> simple liveness check
    POST /predict          -> body: {"url": "..."}  -> verdict + confidence + feature breakdown

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Expects the trained model files at ./model/phishing_model.pkl and ./model/scaler.pkl
(the two files your notebook saved with joblib.dump).
"""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from feature_extractor import PhishingFeatureExtractor

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "phishing_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

app = FastAPI(title="Phishing Website Detector API")

# Allow the static frontend (served from a different port/file://) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None


@app.on_event("startup")
def load_model():
    global model, scaler
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise RuntimeError(
            f"Model files not found. Copy phishing_model.pkl and scaler.pkl "
            f"(from your notebook's joblib.dump step) into {MODEL_DIR}/"
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


class URLRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty")
        return v


class PredictionResponse(BaseModel):
    url: str
    verdict: str          # "phishing" | "legitimate"
    confidence: float      # 0-1, model's confidence in the verdict
    phishing_probability: float
    legitimate_probability: float
    features: dict
    warnings: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: URLRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    extractor = PhishingFeatureExtractor(req.url)
    features, warnings = extractor.extract()

    # Order values exactly as FEATURE_ORDER / the training column order
    ordered_values = [features[name] for name in PhishingFeatureExtractor.FEATURE_ORDER]
    X = np.array(ordered_values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]  # 1 = legitimate, 0 = phishing (per training encoding)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        legit_prob = float(proba[1])
        phishing_prob = float(proba[0])
    else:
        legit_prob = float(pred)
        phishing_prob = 1.0 - legit_prob

    verdict = "legitimate" if pred == 1 else "phishing"
    confidence = legit_prob if pred == 1 else phishing_prob

    return PredictionResponse(
        url=extractor.final_url,
        verdict=verdict,
        confidence=round(confidence, 4),
        phishing_probability=round(phishing_prob, 4),
        legitimate_probability=round(legit_prob, 4),
        features=features,
        warnings=warnings,
    )
