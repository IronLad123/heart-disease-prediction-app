"""
HeartGuard Pro - Production REST API Service
FastAPI backend for integrating Heart Disease Risk Prediction into EHR / web systems.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(
    title="HeartGuard Pro API",
    description="Clinical ML REST API for Cardiac Risk Prediction",
    version="2.0.0"
)

# Load ML artifacts
try:
    model = joblib.load('heart_disease_knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Gender (1 = Male, 0 = Female)")
    cp: int = Field(..., ge=1, le=4, description="Chest Pain Type (1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic)")
    trestbps: float = Field(..., ge=70, le=240, description="Resting Blood Pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=650, description="Serum Cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG Results (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy)")
    thalach: float = Field(..., ge=60, le=230, description="Maximum Heart Rate Achieved (bpm)")
    exang: int = Field(..., ge=0, le=1, description="Exercise Induced Angina (1 = Yes, 0 = No)")
    oldpeak: float = Field(..., ge=0.0, le=7.0, description="ST Depression Induced by Exercise (mm)")
    slope: int = Field(..., ge=1, le=3, description="Slope of Peak Exercise ST Segment (1=Upsloping, 2=Flat, 3=Downsloping)")
    ca: int = Field(..., ge=0, le=3, description="Major Vessels Colored by Fluoroscopy (0-3)")
    thal: int = Field(..., description="Thalassemia (3=Normal, 6=Fixed Defect, 7=Reversible Defect)")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "HeartGuard Pro API",
        "version": "2.0.0",
        "model_loaded": model_loaded
    }

@app.get("/health")
def health_check():
    if not model_loaded:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {load_error}")
    return {"status": "healthy", "accuracy": metadata.get("accuracy", 0.885)}

@app.post("/predict")
def predict_risk(patient: PatientData):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="ML model is not available")
    
    input_df = pd.DataFrame([patient.dict()])
    scaled_df = scaler.transform(input_df)
    
    probability = float(model.predict_proba(scaled_df)[0][1] * 100)
    prediction = int(model.predict(scaled_df)[0])
    
    risk_level = "HIGH RISK" if probability >= 70 else "MODERATE RISK" if probability >= 35 else "LOW RISK"
    
    return {
        "heart_disease_probability": round(probability, 2),
        "prediction": prediction,
        "prediction_label": "Heart Disease Present" if prediction == 1 else "No Heart Disease Detected",
        "risk_level": risk_level,
        "features_processed": patient.dict()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
