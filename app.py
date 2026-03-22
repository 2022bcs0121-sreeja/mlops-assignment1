import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("model.pkl")

class CustomerFeatures(BaseModel):
    ticket_7d: int
    ticket_30d: int
    ticket_90d: int
    sentiment_score: float
    monthly_change: float

@app.post("/predict-risk")
def predict_risk(data: CustomerFeatures):
    features = np.array([[ 
        data.ticket_7d,
        data.ticket_30d,
        data.ticket_90d,
        data.sentiment_score,
        data.monthly_change
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    risk = "HIGH" if prediction == 1 else "LOW"

    return {
        "risk_category": risk,
        "churn_probability": float(probability)
    }