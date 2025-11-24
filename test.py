import requests
import pandas as pd
import time

# Load your dataset
df = pd.read_csv("script_dataset.csv")

url = "http://127.0.0.1:8000/predict"

results = []

for idx, row in df.iterrows():
    payload = {
        "features": row.to_dict() 
    }
    response = requests.post(url, json=payload)
    try:
        pred = response.json()
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        continue

    print(f"Row {idx}: {pred}")
    results.append(pred)
    time.sleep(0.05)
    
results_df = pd.DataFrame(results)
results_df.to_csv("prediction_results.csv", index=False)

print("\nSaved predictions to prediction_results.csv")
