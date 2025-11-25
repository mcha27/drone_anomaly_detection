import os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

path = os.path.abspath('..') + '/' + "model/anomaly_detection_pipeline.pkl"
pipeline = joblib.load(path)

class DroneTraffic(BaseModel):
    features: dict  

@app.post("/predict")
def predict(data: DroneTraffic):
    import pandas as pd
    df = pd.DataFrame([data.features])

    pred_proba = pipeline.predict_proba(df)[0][1]
    pred = int(pred_proba > 0.5)

    return {
        "prediction": pred,
        "probability_malicious": float(pred_proba)
    }
