import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
import umap  # External: umap-learn

from sklearn.preprocessing import StandardScaler, LabelEncoder

def train_and_save_umap(dataset_path: str, n_components: int, n_neighbors: int, pkl_filename: str):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    model, scaler, feature_columns, encoders = None, None, None, {}
    is_trained = False

    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing UMAP model '{pkl_filename}'...**")
        try:
            model, scaler, feature_columns, encoders = load(full_path) 
            print("[INFO] Model loaded successfully.")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Load failed: {e}. Retraining...")

    if not is_trained:
        # --- DATA LOADING (Standard Template) ---
        ext = os.path.splitext(dataset_path)[1].lower()
        try:
            if ext == ".csv": data = pd.read_csv(dataset_path, header = None)
            elif ext in [".xls", ".xlsx"]: data = pd.read_excel(dataset_path)
            elif ext == ".json": data = pd.read_json(dataset_path)
            elif ext == ".parquet": data = pd.read_parquet(dataset_path)
            else: data = pd.read_csv(dataset_path, sep=None, engine='python')
        except Exception as e:
            print(f"[ERROR] Load failed: {e}", file=sys.stderr); sys.exit(1)

        # --- PREPROCESSING ---
        for col in data.columns:
            if data[col].dtype == object:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
        
        data.fillna(data.mean(numeric_only=True), inplace=True) 
        X_df = data.select_dtypes(include=[np.number])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        feature_columns = list(X_df.columns)

        # --- TRAIN UMAP ---
        print(f"\n**[INFO] Training UMAP (components={n_components}, neighbors={n_neighbors})...**")
        model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            random_state=42
        )
        model.fit(X_scaled)
        
        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] UMAP model saved as '{pkl_filename}'")

    def transform_loop():
        nonlocal model, scaler, feature_columns, encoders
        print("\n" + "="*70 + f"\nUMAP TRANSFORMATION MODE ({model.n_components}D)\n" + "="*70)
        while True:
            user_vals = []
            should_exit = False
            for col in feature_columns:
                val = input(f"Enter value for **{col}** (or 'exit'): ")
                if val.lower() in ['exit', 'quit']: should_exit = True; break
                try:
                    if col in encoders:
                        user_vals.append(encoders[col].transform([val.strip()])[0])
                    else:
                        user_vals.append(float(val))
                except: print("[ERROR] Invalid input."); break
            
            if should_exit: break
            if len(user_vals) == len(feature_columns):
                df_input = pd.DataFrame([user_vals], columns=feature_columns)
                user_scaled = scaler.transform(df_input)
                reduced = model.transform(user_scaled)
                print(f"\n**[RESULT] UMAP Vector:**\n{reduced[0]}\n" + "-"*30)

    return transform_loop

if __name__ == "__main__":
    if len(sys.argv) == 5:
        func = train_and_save_umap(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        func()
    elif len(sys.argv) == 3 and sys.argv[2] == 'transform_only':
        func = train_and_save_umap(None, 0, 0, sys.argv[1])
        func()