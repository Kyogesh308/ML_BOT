import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.manifold import SpectralEmbedding


def train_and_save_spectral_embedding(dataset_path, n_components, pkl_filename):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    if os.path.exists(full_path):
        print(f"SpectralEmbedding model already trained. Loading '{pkl_filename}'...")
        se_model, X_embedded, columns = load(full_path)

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

        se_model = SpectralEmbedding(
            n_components=n_components,
            affinity="nearest_neighbors",
            random_state=42
        )

        X_embedded = se_model.fit_transform(data)
        print(f"Training complete. Reduced shape: {X_embedded.shape}")

        dump((se_model, X_embedded, list(data.columns)), full_path)
        print(f"SpectralEmbedding model saved as '{pkl_filename}'")

    def transform_input():
        _, X_embedded, columns = load(full_path)

        print("\nSpectralEmbedding does not support transforming new samples.")
        print("Showing saved reduced representation instead:\n")

        print(X_embedded)

    return transform_input


if __name__ == "__main__":
    predict_func = train_and_save_spectral_embedding(
        dataset_path=sys.argv[1],
        n_components=int(sys.argv[2]),
        pkl_filename=sys.argv[3],
    )
    predict_func()
