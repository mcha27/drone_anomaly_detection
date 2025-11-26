import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List

app = FastAPI()

path = os.path.abspath('..') + '/' + "models/kmeans_model.pkl"
pipeline = joblib.load(path)

class DroneTraffic(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(data: DroneTraffic):
    features_array = np.array(data.features).reshape(1, -1)
    labels = pipeline.predict(features_array)
    centers = pipeline.cluster_centers_
    distances = np.linalg.norm(features_array - centers[labels], axis=1)
    threshold = 7.83 #np.percentile(distances, 95)
    anomaly = 1 if distances[0] > threshold else 0
    return {
        "prediction": anomaly
    }
