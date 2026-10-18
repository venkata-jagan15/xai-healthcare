from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add the parent directory to sys.path to import XAI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xai.shap_explainer import ShapExplainer
from xai.lime_explainer import LimeExplainer

app = FastAPI(title="XAI Healthcare Risk Prediction")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = "xai-healthcare/backend/model.pkl"
SCALER_PATH = "xai-healthcare/backend/scaler.pkl"
FEATURE_NAMES_PATH = "xai-healthcare/backend/feature_names.pkl"
TRAIN_DATA_PATH = "xai-healthcare/data/heart.csv"

# Global variables for model and explainers
model = None
scaler = None
feature_names = None
shap_explainer = None
lime_explainer = None

def load_resources():
    global model, scaler, feature_names, shap_explainer, lime_explainer
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        shap_explainer = ShapExplainer(MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
        lime_explainer = LimeExplainer(MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH, TRAIN_DATA_PATH)
        print("Resources loaded successfully.")
    else:
        print("Model artifacts not found. Please run train_model.py first.")

@app.on_event("startup")
async def startup_event():
    load_resources()

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def read_root():
    return {"message": "XAI Healthcare API is running"}

@app.post("/predict")
async def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert input to DataFrame
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Prediction
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = int(model.predict(input_scaled)[0])
    
    # SHAP Explanation
    shap_result = shap_explainer.explain(input_dict)
    
    # LIME Explanation
    lime_result = lime_explainer.explain(input_dict)
    
    # Comparative Analysis (Consistency Score)
    # Get top 5 features from SHAP
    shap_vals = np.abs(shap_result["shap_values"])
    top_shap_indices = np.argsort(shap_vals)[-5:]
    top_shap_features = [feature_names[i] for i in top_shap_indices]
    
    # Get top 5 features from LIME
    lime_features = [item[0].split(' <= ')[0].split(' > ')[0].strip() for item in lime_result["exp_list"][:5]]
    # Clean lime features (some might have comparison operators)
    cleaned_lime_features = []
    for f in lime_features:
        for original in feature_names:
            if original in f:
                cleaned_lime_features.append(original)
                break
    
    common_features = set(top_shap_features).intersection(set(cleaned_lime_features))
    consistency_score = (len(common_features) / 5) * 100 if len(top_shap_features) > 0 else 0
    
    return {
        "prediction": prediction,
        "probability": float(prob),
        "risk_level": "High Risk" if prob > 0.7 else ("Medium Risk" if prob > 0.3 else "Low Risk"),
        "shap": shap_result,
        "lime": lime_result,
        "consistency_score": consistency_score,
        "common_features": list(common_features)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
