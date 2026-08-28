import sys
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# Ensure Vercel finds local modules properly
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR))

try:
    from feature_extractor import PhishingFeatureExtractor
except ImportError:
    from backend.feature_extractor import PhishingFeatureExtractor

MODEL_DIR = CURRENT_DIR / "model"
MODEL_PATH = MODEL_DIR / "phishing_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

app = FastAPI(title="Phishing Website Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None

def get_model():
    """Lazy loading model and scaler to optimize Vercel serverless functions"""
    global model, scaler
    if model is None or scaler is None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise RuntimeError("Model or Scaler file not found in backend/model directory.")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    return model, scaler

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
    verdict: str
    confidence: float
    phishing_probability: float
    legitimate_probability: float
    features: dict
    warnings: list[str]

@app.get("/")
@app.get("/api/health")
@app.get("/health")
def health():
    try:
        m, _ = get_model()
        return {"status": "ok", "model_loaded": m is not None}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: URLRequest):
    try:
        mdl, scl = get_model()

        extractor = PhishingFeatureExtractor(req.url)
        features, warnings = extractor.extract()

        ordered_values = [features[name] for name in PhishingFeatureExtractor.FEATURE_ORDER]
        X = np.array(ordered_values).reshape(1, -1)
        X_scaled = scl.transform(X)

        pred = mdl.predict(X_scaled)[0]

        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X_scaled)[0]
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature extraction or prediction failed: {str(e)}")