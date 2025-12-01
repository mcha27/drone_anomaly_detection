import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import joblib

path = os.path.abspath('..') + '/' + "datasets/model_dataset.csv"
df = pd.read_csv(path)

attack_cols = ['label_spoofing', 'label_mitm', 'label_ddos', 'label_gps_spoofing', 'label_malware', 'label_jamming', 'label_protocol_exploit']
df = df.drop(['timestamp', 'drone_gps_coordinates', 'label_normal'], axis=1)
features = df.columns.to_numpy()
features = np.delete(features, -1)

X = df[features]
y = df[attack_cols].values

categorical_cols = ["communication_protocol", "encryption_type"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("pca", PCA(n_components=0.95)),
    ("model", XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    ))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "anomaly_detection_pipeline.pkl")

print("Pipeline saved successfully.")
