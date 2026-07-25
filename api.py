"""
HeartGuard AI - Next-Gen Multi-Model REST API Service
FastAPI backend for querying multi-model ensemble suite predictions programmatically.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

app = FastAPI(
    title="HeartGuard AI Multi-Model REST API",
    description="Production ML API serving 5 Multi-Model Cardiac Classifiers (Random Forest, Gradient Boosting, KNN, Logistic Regression, Voting Ensemble)",
    version="3.0.0"
)

# Load ML Suite with dynamic fallback
try:
    with open('models_metadata.json', 'r') as f:
        metadata = json.load(f)
    scaler = joblib.load('scaler.pkl')
    
    models = {}
    for m_name, info in metadata['models'].items():
        models[m_name] = joblib.load(info['filename'])
    models_loaded = True
except Exception as e:
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        try:
            df = pd.read_csv(url, names=column_names, na_values='?')
        except Exception:
            df = pd.read_csv('Heart Disease Data/processed.cleveland.data', names=column_names, na_values='?')
            
        df = df.dropna().reset_index(drop=True)
        df['target'] = (df['target'] > 0).astype(int)
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled, y),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_scaled, y),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7).fit(X_scaled, y),
            'Logistic Regression': LogisticRegression(random_state=42).fit(X_scaled, y)
        }
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', models['Random Forest']),
                ('gb', models['Gradient Boosting']),
                ('knn', models['K-Nearest Neighbors']),
                ('lr', models['Logistic Regression'])
            ],
            voting='soft'
        ).fit(X_scaled, y)
        
        models['Voting Ensemble'] = ensemble
        models_loaded = True
    except Exception as ex:
        models_loaded = False
        load_error = str(ex)

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
        "service": "HeartGuard AI Multi-Model REST API",
        "version": "3.0.0",
        "available_models": list(models.keys()) if models_loaded else [],
        "models_loaded": models_loaded
    }

@app.get("/health")
def health_check():
    if not models_loaded:
        raise HTTPException(status_code=500, detail=f"Models failed to load: {load_error}")
    return {"status": "healthy", "available_models": list(models.keys())}

@app.post("/predict")
def predict_risk(
    patient: PatientData, 
    model_name: Optional[str] = Query("Voting Ensemble", description="ML Model: 'Random Forest', 'Gradient Boosting', 'K-Nearest Neighbors', 'Logistic Regression', 'Voting Ensemble'")
):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="ML model suite is not available")
    
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Invalid model_name '{model_name}'. Choose from: {list(models.keys())}")

    target_model = models[model_name]
    input_df = pd.DataFrame([patient.dict()])
    scaled_df = scaler.transform(input_df)
    
    probability = float(target_model.predict_proba(scaled_df)[0][1] * 100)
    prediction = int(target_model.predict(scaled_df)[0])
    
    risk_level = "HIGH RISK" if probability >= 70 else "MODERATE RISK" if probability >= 35 else "LOW RISK"
    
    return {
        "model_used": model_name,
        "heart_disease_probability": round(probability, 2),
        "prediction": prediction,
        "prediction_label": "Heart Disease Present" if prediction == 1 else "No Heart Disease Detected",
        "risk_level": risk_level,
        "features_processed": patient.dict()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
