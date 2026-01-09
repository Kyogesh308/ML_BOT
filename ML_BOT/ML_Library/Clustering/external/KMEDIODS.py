import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn_extra.cluster import KMedoids # External: scikit-learn-extra
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score

def train_and_save_kmedoids(dataset_path: str, n_clusters: int, pkl_filename: str):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    model, scaler, feature_columns, encoders = None, None, None, {}
    is_trained = False

    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing K-Medoids model '{pkl_filename}'...**")
        try:
            model, scaler, feature_columns, encoders = load(full_path) 
            print("[INFO] Model loaded successfully.")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Load failed: {e}. Retraining...")

    if not is_trained:
        # --- DATA LOADING ---
        ext = os.path.splitext(dataset_path)[1].lower()
        try:
            if ext == ".csv": data = pd.read_csv(dataset_path)
            elif ext in [".xls", ".xlsx"]: data = pd.read_excel(dataset_path)
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

        # --- TRAIN K-MEDOIDS ---
        print(f"\n**[INFO] Training K-Medoids (n_clusters={n_clusters})...**")
        model = KMedoids(n_clusters=n_clusters, random_state=42, method='pam')
        labels = model.fit_predict(X_scaled)
        
        # Evaluation
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            print(f"[INFO] Silhouette Score: {score:.4f}")

        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] K-Medoids model saved as '{pkl_filename}'")

    def predict_loop():
        nonlocal model, scaler, feature_columns, encoders
        print("\n" + "="*70 + f"\nK-MEDOIDS PREDICTION MODE ({model.n_clusters} Clusters)\n" + "="*70)
        while True:
            user_vals = []
            should_exit = False
            for col in feature_columns:
                val = input(f"Enter {col}: ")
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
                cluster = model.predict(user_scaled)[0]
                print(f"\n**[RESULT] Assigned Cluster: {cluster}**\n" + "-"*30)

    return predict_loop

if __name__ == "__main__":
    if len(sys.argv) == 4:
        func = train_and_save_kmedoids(sys.argv[1], int(sys.argv[2]), sys.argv[3])
        func()
    elif len(sys.argv) == 3 and sys.argv[2] == 'predict_only':
        func = train_and_save_kmedoids(None, 0, sys.argv[1])
        func()