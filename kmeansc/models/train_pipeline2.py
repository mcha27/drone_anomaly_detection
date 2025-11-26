import os
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

path = os.path.abspath("..") + '/' + "datasets/model_dataset.csv"
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

k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_pca)

joblib.dump(kmeans, 'kmeans_model.pkl')