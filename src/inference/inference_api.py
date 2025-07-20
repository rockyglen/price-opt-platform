import os
import joblib
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum

# === Load environment variables ===
load_dotenv()

MODEL_PATH = "src/model/latest_model.joblib"
FEATURE_ORDER_PATH = "src/model/feature_order.pkl"

# === FastAPI App ===
app = FastAPI(title="Pricing ML Inference API")

# === Enum Definitions ===
class Category(str, Enum):
    electronics = "electronics"
    books = "books"
    clothing = "clothing"
    home = "home"
    beauty = "beauty"

class Segment(str, Enum):
    deal_seeker = "deal-seeker"
    premium = "premium"
    loyal = "loyal"
    new = "new"
    impulsive = "impulsive"

class TimeOfDay(str, Enum):
    morning = "morning"
    afternoon = "afternoon"
    evening = "evening"
    night = "night"

class DayOfWeek(str, Enum):
    mon = "mon"
    tue = "tue"
    wed = "wed"
    thu = "thu"
    fri = "fri"
    sat = "sat"
    sun = "sun"

class Season(str, Enum):
    peak = "peak"
    off = "off"

# === Inference Input Schema ===
class InferenceRequest(BaseModel):
    offered_price: float
    base_price: float
    competitor_price: float
    category: Category
    segment: Segment
    time_of_day: TimeOfDay
    day_of_week: DayOfWeek
    season: Season

# === Load Feature Order ===
if not os.path.exists(FEATURE_ORDER_PATH):
    raise FileNotFoundError(f"❌ Feature order file not found at: {FEATURE_ORDER_PATH}")
feature_order = joblib.load(FEATURE_ORDER_PATH)

# === Load Model ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Trained model not found at: {MODEL_PATH}")
try:
    print("🔁 Loading model from local joblib...")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully from joblib!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# === Preprocessing Logic ===
def preprocess(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])

    categoricals = {
        "category": [e.value for e in Category],
        "segment": [e.value for e in Segment],
        "time_of_day": [e.value for e in TimeOfDay],
        "day_of_week": [e.value for e in DayOfWeek],
        "season": [e.value for e in Season],
    }

    for col, values in categoricals.items():
        for val in values:
            df[f"{col}_{val}"] = 1 if input_dict[col] == val else 0
        df.drop(columns=[col], inplace=True)

    df = df.reindex(columns=feature_order, fill_value=0)
    return df

# === Prediction Endpoint ===
@app.post("/predict")
def predict(request: InferenceRequest):
    try:
        input_df = preprocess(request.dict())
        proba = model.predict_proba(input_df)[0][1]
        return {"converted_probability": round(float(proba), 4)}
    except Exception as e:
        return {"error": f"❌ Prediction failed: {str(e)}"}

# === Run Server Locally ===
if __name__ == "__main__":
    uvicorn.run("src.inference.inference_api:app", host="0.0.0.0", port=8000, reload=True)
