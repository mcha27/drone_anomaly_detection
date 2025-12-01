import os
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

path = os.path.abspath("..") + '/' + "datasets/model_dataset.csv"
df = pd.read_csv(path)

path = os.path.abspath('..') + '/' + "datasets/model_dataset.csv"
df = pd.read_csv(path)
    
attack_cols = [
    'label_spoofing', 'label_mitm', 'label_ddos',
    'label_gps_spoofing', 'label_malware',
    'label_jamming', 'label_protocol_exploit'
]

df['anomaly_true'] = df[attack_cols].max(axis=1)

df = df.drop(['timestamp', 'drone_gps_coordinates', 'label_normal'], axis=1)
df = df.drop(columns=attack_cols)

categorical_cols = ['communication_protocol', 'encryption_type']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_cols)
)

df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

y_true = df['anomaly_true'].values

df = df.drop(columns=['anomaly_true'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

k = 8  # normal vs anomaly
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_pca)

joblib.dump(kmeans, 'kmeans_model.pkl')
print("Model saved successfully.")