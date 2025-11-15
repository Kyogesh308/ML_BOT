import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.decomposition import IncrementalPCA


def train_and_save_incremental_pca(dataset_path, n_components, batch_size, pkl_filename):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    if os.path.exists(full_path):
        print(f"IncrementalPCA model already trained. Loading '{pkl_filename}'...")
        ipca_model, columns = load(full_path)

    else:
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
                data = pd.read_csv(dataset_path, sep=None, engine="python")
        except:
            with open(dataset_path, "r", encoding="utf-8") as f:
                content = f.read()
            data = pd.read_csv(StringIO(content))

        data = data.select_dtypes(include=[np.number]).fillna(0)
        columns = list(data.columns)
        X = data.to_numpy()

        ipca_model = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size
        )

        # Partial fitting loop
        for i in range(0, X.shape[0], batch_size):
            batch = X[i:i + batch_size]
            ipca_model.partial_fit(batch)

        X_reduced = ipca_model.transform(X)
        print(f"Training complete. Reduced shape: {X_reduced.shape}")

        dump((ipca_model, columns), full_path)
        print(f"IncrementalPCA model saved as '{pkl_filename}'")

    def transform_input():
        ipca_model, columns = load(full_path)

        X_new = []
        print("\nEnter values for the following columns:")
        for col in columns:
            val = float(input(f"{col}: "))
            X_new.append(val)

        reduced = ipca_model.transform([X_new])
        print("\nIncrementalPCA Reduced Representation:")
        print(reduced[0])

    return transform_input


if __name__ == "__main__":
    predict_func = train_and_save_incremental_pca(
        dataset_path=sys.argv[1],
        n_components=int(sys.argv[2]),
        batch_size=int(sys.argv[3]),
        pkl_filename=sys.argv[4],
    )
    predict_func()
