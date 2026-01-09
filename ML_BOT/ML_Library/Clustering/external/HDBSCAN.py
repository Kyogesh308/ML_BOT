import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
import hdbscan # External library
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score

def train_and_save_hdbscan(dataset_path: str, min_cluster_size: int, pkl_filename: str):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    model, scaler, feature_columns, encoders = None, None, None, {}
    is_trained = False

    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing HDBSCAN model '{pkl_filename}'...**")
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
            data = pd.read_csv(dataset_path) if ext == ".csv" else pd.read_excel(dataset_path)
        except: data = pd.read_csv(dataset_path, sep=None, engine='python')

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

        # --- TRAIN HDBSCAN ---
        print(f"\n**[INFO] Training HDBSCAN (min_cluster_size={min_cluster_size})...**")
        # prediction_data=True is required to enable .predict() for new points later
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
        model.fit(X_scaled)
        
        labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[INFO] Found {n_clusters} clusters.")

        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] HDBSCAN model saved.")

    def predict_loop():
        nonlocal model, scaler, feature_columns, encoders
        print("\n" + "="*70 + "\nHDBSCAN PREDICTION MODE (Detects Noise as -1)\n" + "="*70)
        while True:
            user_vals = []
            should_exit = False
            for col in feature_columns:
                val = input(f"Enter {col}: ")
                if val.lower() in ['exit', 'quit']: should_exit = True; break
                try:
                    user_vals.append(encoders[col].transform([val.strip()])[0] if col in encoders else float(val))
                except: print("[ERROR] Invalid input."); break
            
            if should_exit: break
            if len(user_vals) == len(feature_columns):
                df_input = pd.DataFrame([user_vals], columns=feature_columns)
                user_scaled = scaler.transform(df_input)
                # HDBSCAN approx_predict is used for new unseen data
                cluster, strength = hdbscan.approximate_predict(model, user_scaled)
                print(f"\n**[RESULT] Cluster: {cluster[0]} (Confidence: {strength[0]:.2f})**\n" + "-"*30)

    return predict_loop

if __name__ == "__main__":
    if len(sys.argv) == 4:
        func = train_and_save_hdbscan(sys.argv[1], int(sys.argv[2]), sys.argv[3])
        func()
    elif len(sys.argv) == 3 and sys.argv[2] == 'predict_only':
        func = train_and_save_hdbscan(None, 0, sys.argv[1])
        func()