import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import time

pipeline = joblib.load("kmeans_model.pkl")

path = os.path.abspath("..") + '/' + "datasets/script_dataset.csv"
df = pd.read_csv(path)

attack_cols = ['label_spoofing', 'label_mitm', 'label_ddos', 'label_gps_spoofing',
               'label_gps_spoofing', 'label_malware', 'label_jamming', 'label_protocol_exploit']

df = df.drop(['timestamp', 'drone_gps_coordinates', 'label_normal'], axis=1)
df['anomaly'] = df[attack_cols].max(axis=1)
df = df.drop(columns=attack_cols)
df = df.drop(columns=['anomaly'])

categorical = ['communication_protocol', 'encryption_type']
numeric = [col for col in df.columns if col not in categorical]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['communication_protocol', 'encryption_type']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['communication_protocol', 'encryption_type']))
df = pd.concat([df.drop(['communication_protocol', 'encryption_type'], axis=1), encoded_df], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

labels = pipeline.predict(X_pca)
centers = pipeline.cluster_centers_
distances = np.linalg.norm(X_pca - centers[labels], axis=1)
print(len(distances))
threshold = np.percentile(distances, 95)
print(threshold)
#for dis in distances:
#    anomaly = 1 if dis > threshold else 0
#    print(anomaly)
#    time.sleep(0.05)
