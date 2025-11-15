import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.manifold import Isomap
from sklearn.datasets import MNIST

def train_and_save_isomap(dataset_path, n_components, n_neighbors, pkl_filename):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    if os.path.exists(full_path):
        print(f"Isomap model already trained. Loading '{pkl_filename}'...")
        isomap_model, columns = load(full_path)

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
                data = pd.read_csv(dataset_path, sep=None, engine='python')
        except:
            with open(dataset_path, "r", encoding="utf-8") as f:
                content = f.read()
            data = pd.read_csv(StringIO(content))

        data = data.select_dtypes(include=[np.number]).fillna(0)

        isomap_model = Isomap(
            n_neighbors=n_neighbors,
            n_components=n_components
        )

        X_reduced = isomap_model.fit_transform(data)
        print(f"Training complete. Reduced shape: {X_reduced.shape}")

        dump((isomap_model, list(data.columns)), full_path)
        print(f"Isomap model saved as '{pkl_filename}'")

    def transform_input():
        isomap_model, columns = load(full_path)

        X_new = []
        print("\nEnter values for the following columns:")
        for col in columns:
            val = float(input(f"{col}: "))
            X_new.append(val)

        reduced = isomap_model.transform([X_new])
        print("\nIsomap Reduced Representation:")
        print(reduced[0])

    return transform_input


if __name__ == "__main__":
    predict_func = train_and_save_isomap(
        dataset_path=sys.argv[1],
        n_components=int(sys.argv[2]),
        n_neighbors=int(sys.argv[3]),
        pkl_filename=sys.argv[4],
    )
    predict_func()
