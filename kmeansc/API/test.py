import requests
import pandas as pd
import time
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

path = os.path.abspath('..') + '/' + "datasets/script_dataset.csv"
df = pd.read_csv(path)

attack_cols = ['label_spoofing', 'label_mitm', 'label_ddos', 'label_gps_spoofing',
               'label_gps_spoofing', 'label_malware', 'label_jamming', 'label_protocol_exploit']
df = df.drop(['timestamp', 'drone_gps_coordinates', 'label_normal'], axis=1)
df['anomaly'] = df[attack_cols].max(axis=1)
df = df.drop(columns=attack_cols)
df = df.drop(columns=['anomaly'])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['communication_protocol', 'encryption_type']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['communication_protocol', 'encryption_type']))
df = pd.concat([df.drop(['communication_protocol', 'encryption_type'], axis=1), encoded_df], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

url = "http://127.0.0.1:8000/predict"

results = []
for idx, row in enumerate(X_pca):
    payload = {
        "features": row.tolist()  # Convert NumPy row to list
    }
    response = requests.post(url, json=payload)
    try:
        pred = response.json()
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        continue
    print(f"Row {idx}: {pred}")
    if pred == 1:
        results.append([idx, pred])
    time.sleep(0.05) # spaced

rows_with_ones = [row for row in results if row[1] == 1]

# Count how many 1s
count_ones = len(rows_with_ones)

# Create DataFrame
df_output = pd.DataFrame(rows_with_ones, columns=["row_index", "label"])

# Add count row on top
df_output.loc[-1] = ["COUNT", count_ones]
df_output.index = df_output.index + 1
df_output = df_output.sort_index()

# Save to CSV
df_output.to_csv("rows_with_ones_output.csv", index=False)

print("Saved rows_with_ones_output.csv")
np.savetxt("kmeans_results.csv", rows_with_ones, delimiter=",", fmt="%d")

print("\nSaved predictions to kmeans_results.csv")