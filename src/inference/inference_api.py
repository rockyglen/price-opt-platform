from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import joblib
import uvicorn
import os

app = FastAPI(title="Pricing ML Inference API")

# --- Load feature order ---
FEATURE_ORDER_PATH = "src/model/feature_order.pkl"
if not os.path.exists(FEATURE_ORDER_PATH):
    raise FileNotFoundError(f"Feature order file not found at {FEATURE_ORDER_PATH}")

feature_order = joblib.load(FEATURE_ORDER_PATH)

# --- Define categorical levels ---
CATEGORICALS = {
    "category": ["electronics", "books", "clothing", "home", "beauty"],
    "segment": ["deal-seeker", "premium", "loyal", "new", "impulsive"],
    "time_of_day": ["morning", "afternoon", "evening", "night"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
    "season": ["peak", "off"],
}

# --- Inference request format ---
class InferenceRequest(BaseModel):
    offered_price: float
    base_price: float
    competitor_price: float
    category: str
    segment: str
    time_of_day: str
    day_of_week: str
    season: str

# --- Load MLflow model ---
mlflow.set_tracking_uri("file:///Users/glenlouis/Coding💻/price-opt-platform/mlruns")
MODEL_URI = "models:/pricing_ml_model/4"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model from {MODEL_URI}: {e}")

# --- Preprocessing logic ---
def preprocess(input_dict):
    df = pd.DataFrame([input_dict])

    # One-hot encode manually
    for col, categories in CATEGORICALS.items():
        for cat in categories:
            df[f"{col}_{cat}"] = 1 if input_dict[col] == cat else 0
        df.drop(columns=[col], inplace=True)

    # Reindex to match training time
    df = df.reindex(columns=feature_order, fill_value=0)
    return df

# --- Inference Endpoint ---
@app.post("/predict")
def predict(request: InferenceRequest):
    try:
        input_df = preprocess(request.dict())
        probability = model.predict_proba(input_df)[0][1]
        return {"converted_probability": round(float(probability), 4)}
    except Exception as e:
        return {"error": str(e)}

# --- Entry point ---
if __name__ == "__main__":
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=True)
