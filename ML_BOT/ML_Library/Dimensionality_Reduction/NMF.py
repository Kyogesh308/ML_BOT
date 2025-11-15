import pandas as pd
import numpy as np
import os
import sys
import importlib
from io import StringIO
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler


def train_and_save_nmf(dataset_path, n_components, pkl_filename):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)

    full_path = os.path.join(trained_dir, pkl_filename)

    # -------------------------------------------------
    # Load existing model
    # -------------------------------------------------
    if os.path.exists(full_path):
        print(f"NMF reducer already trained. Loading '{pkl_filename}'...")
        nmf_model, scaler, columns = load(full_path)
        print("Loaded successfully.")
    else:
        # -------------------------------------------------
        # Load dataset (multi-format support)
        # -------------------------------------------------
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
        except:
            with open(dataset_path, "r", encoding="utf-8") as f:
                content = f.read()
            data = pd.read_csv(StringIO(content))

        # -------------------------------------------------
        # NMF requires numeric and non-negative data
        # -------------------------------------------------
        data = data.select_dtypes(include=[np.number])
        if data.shape[1] == 0:
            raise ValueError("Dataset contains no numeric columns.")

        data.fillna(data.mean(), inplace=True)

        # -------------------------------------------------
        # Scale to non-negative range [0, 1]
        # MinMaxScaler is the correct preprocessing for NMF
        # -------------------------------------------------
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)

        # -------------------------------------------------
        # Import NMF dynamically
        # -------------------------------------------------
        from sklearn.decomposition import NMF
        nmf_model = NMF(n_components=n_components, init="nndsvda", max_iter=1000)

        # -------------------------------------------------
        # Fit NMF
        # -------------------------------------------------
        X_reduced = nmf_model.fit_transform(X_scaled)
        print(f"Training complete. Reduced shape: {X_reduced.shape}")

        # -------------------------------------------------
        # Save the reducer
        # -------------------------------------------------
        dump((nmf_model, scaler, list(data.columns)), full_path)
        print(f"NMF reducer saved as '{pkl_filename}'")

    # -------------------------------------------------
    # Prediction / Transform function
    # -------------------------------------------------
    def transform_input():
        nmf_model, scaler, columns = load(full_path)

        X_new = []
        print("\nEnter values for the following columns:")
        for col in columns:
            val = float(input(f"{col}: "))
            X_new.append(val)

        X_scaled = scaler.transform([X_new])
        reduced = nmf_model.transform(X_scaled)

        print("\nNMF Reduced Representation:")
        print(reduced[0])

    return transform_input


if __name__ == "__main__":
    predict_func = train_and_save_nmf(
        dataset_path=sys.argv[1],
        n_components=int(sys.argv[2]),
        pkl_filename=sys.argv[3],
    )
    predict_func()
