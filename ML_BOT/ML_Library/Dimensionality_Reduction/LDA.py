import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.decomposition import LatentDirichletAllocation


def train_and_save_lda(dataset_path, n_components, pkl_filename):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    if os.path.exists(full_path):
        print(f"LDA model already trained. Loading '{pkl_filename}'...")
        lda_model, columns = load(full_path)
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

        if (data < 0).any().any():
            raise ValueError("LDA requires non-negative integer count data.")

        data = data.round().astype(int)

        lda_model = LatentDirichletAllocation(
            n_components=n_components,
            learning_method="batch",
            max_iter=50,
            random_state=42
        )

        X_reduced = lda_model.fit_transform(data)
        print(f"Training complete. Reduced shape: {X_reduced.shape}")

        dump((lda_model, list(data.columns)), full_path)
        print(f"LDA model saved as '{pkl_filename}'")

    def transform_input():
        lda_model, columns = load(full_path)

        X_new = []
        print("\nEnter integer counts for the following columns:")
        for col in columns:
            val = int(input(f"{col}: "))
            if val < 0:
                raise ValueError("Counts cannot be negative.")
            X_new.append(val)

        reduced = lda_model.transform([X_new])
        print("\nLDA Reduced Representation:")
        print(reduced[0])

    return transform_input


if __name__ == "__main__":
    predict_func = train_and_save_lda(
        dataset_path=sys.argv[1],
        n_components=int(sys.argv[2]),
        pkl_filename=sys.argv[3],
    )
    predict_func()
