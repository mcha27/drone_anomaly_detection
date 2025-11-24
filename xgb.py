import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, RocCurveDisplay
import xgboost as xgb
import joblib

df = pd.read_csv('drone_communication_dataset.csv')

df = df.drop(['label_normal', 'label_spoofing', 'label_mitm', 'label_ddos', 'label_gps_spoofing', 'label_malware', 'label_protocol_exploit'], axis=1)
df = df.drop(['timestamp', 'drone_gps_coordinates'], axis=1)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['communication_protocol', 'encryption_type']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['communication_protocol', 'encryption_type']))
df = pd.concat([df.drop(['communication_protocol', 'encryption_type'], axis=1), encoded_df], axis=1)

labels = df['label_jamming']

legit = df[labels == 0].copy()
jamming = df[labels  == 1].copy()

features = df.columns.to_numpy()
features = np.delete(features, -1)

X = df[features]
y = df['label_jamming'] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

for row in X_pca:
    print(row)

