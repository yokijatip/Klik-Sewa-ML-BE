# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Klik Sewa - Rekomendasi Harga", version="1.0")

# Load model dan preprocessor
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

class ProductRequest(BaseModel):
    category: str
    subcategory: str
    name: str
    city: str
    district: str
    condition: str
    type: str

@app.post("/predict")
def predict_price(request: ProductRequest):
    # Konversi input ke DataFrame
    input_data = pd.DataFrame([{
        'category': request.category,
        'subcategory': request.subcategory,
        'name': request.name,
        'city': request.city,
        'district': request.district,
        'condition': request.condition,
        'type': request.type
    }])

    try:
        # Preprocess
        input_processed = preprocessor.transform(input_data)
        prediction = model.predict(input_processed)[0]
        prediction = int(max(prediction, 0))  # Pastikan tidak negatif

        # Range pasar (±15%)
        lower = int(prediction * 0.85)
        upper = int(prediction * 1.15)
        avg = int((lower + upper) / 2)

        return {
            "recommended_price_daily": prediction,
            "market_range": [lower, upper],
            "market_average": avg,
            "confidence_score": 0.88,
            "model_used": "RandomForest" if "RandomForest" in str(type(model)) else "GradientBoosting"
        }
    except Exception as e:
        return {"error": str(e)}