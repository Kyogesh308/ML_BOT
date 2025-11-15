import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.preprocessing import StandardScaler  # Use MaxAbsScaler for sparse data
from sklearn.decomposition import TruncatedSVD

def train_and_save_reducer(dataset_path, n_components, pkl_filename):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Load existing model if already trained
    if os.path.exists(full_path):
        print(f"Reducer already trained. Loading '{pkl_filename}'...")
        reducer, scaler, columns = load(full_path)
        print("Reducer loaded successfully.")
    else:
        # Load dataset with multi-format support
        ext = os.path.splitext(dataset_path)[1].lower()
        try:
            if ext == ".csv":
                data = pd.read_csv(dataset_path)
            elif ext in [".xls", ".xlsx"]:
                data = pd.read_excel(dataset_path)
            elif ext == ".json":
                data = pd.read_json(dataset_path)
            elif ext == ".parquet":
                data = pd.read_parquet(dataset_path)
            else:
                data = pd.read_csv(dataset_path, sep=None, engine='python')
        except Exception:
            with open(dataset_path, "r", encoding="utf-8") as f:
                content = f.read()
            data = pd.read_csv(StringIO(content))

        # Select numeric columns only
        data = data.select_dtypes(include=[np.number])
        if data.shape[1] == 0:
            raise ValueError("Dataset contains no numeric columns.")

        # Validate n_components
        if n_components >= data.shape[1]:
            raise ValueError(f"n_components ({n_components}) must be less than number of features ({data.shape[1]}).")

        # Handle missing values
        data.fillna(data.mean(), inplace=True)

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Train reducer
        reducer = TruncatedSVD(n_components=n_components)
        X_reduced = reducer.fit_transform(X_scaled)
        print(f"Training complete. Reduced shape: {X_reduced.shape}")

        # Save model
        dump((reducer, scaler, list(data.columns)), full_path)
        print(f"Reducer saved as '{pkl_filename}'")

    # Interactive transformation
    def transform_input():
        reducer, scaler, columns = load(full_path)
        X_new = []
        print("\nEnter values for the following columns:")
        for col in columns:
            val = float(input(f"{col}: "))
            X_new.append(val)

        X_scaled = scaler.transform([X_new])
        reduced = reducer.transform(X_scaled)

        print("\nReduced Vector Representation:")
        print(reduced[0])

    return transform_input

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python svd_reducer.py <dataset_path> <n_components> <pkl_filename>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    n_components = int(sys.argv[2])
    pkl_filename = sys.argv[3]

    predict_func = train_and_save_reducer(dataset_path, n_components, pkl_filename)
    predict_func()