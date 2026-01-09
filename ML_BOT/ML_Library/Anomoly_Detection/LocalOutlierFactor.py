import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder

def train_and_save_lof(dataset_path: str, n_neighbors: int, contamination: float, pkl_filename: str):
    """
    Trains a Local Outlier Factor model for Novelty Detection.
    Note: 'novelty=True' allows the model to predict on new, unseen data.
    """
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    model, scaler, feature_columns, encoders = None, None, None, {}
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing LOF model '{pkl_filename}'...**")
        try:
            model, scaler, feature_columns, encoders = load(full_path) 
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Failed to load: {e}. Retraining...")

    # --- CASE 2: Train a new model ---
    if not is_trained:
        # --- DATA LOADING (Robust Block) ---
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
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}", file=sys.stderr)
            sys.exit(1)
        
        # --- PREPROCESSING ---
        # 1. Encode Categorical Data
        for col in data.columns:
            if data[col].dtype == object:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
        
        # 2. Fill NaNs
        data.fillna(data.mean(numeric_only=True), inplace=True) 
        X_df = data.select_dtypes(include=[np.number])
        
        # 3. Scaling (Essential for neighbors-based algorithms)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        feature_columns = list(X_df.columns)

        # --- TRAIN LOF ---
        print(f"\n**[INFO] Training LOF (n_neighbors={n_neighbors}, contamination={contamination})...**")
        
        # novelty=True is REQUIRED to use .predict() on new data
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors, 
            contamination=contamination, 
            novelty=True,
            n_jobs=-1
        )
        model.fit(X_scaled)
        
        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] Model saved successfully as '{pkl_filename}'")

    # --- THE DETECTION LOOP ---
    def detect_loop():
        nonlocal model, scaler, feature_columns, encoders
        
        if model is None:
            model, scaler, feature_columns, encoders = load(full_path)

        print("\n" + "="*70)
        print("ANOMALY DETECTION MODE: LOCAL OUTLIER FACTOR")
        print("Type 'exit' to quit.")
        print("="*70)

        while True:
            user_vals = []
            should_exit = False
            print("\n" + "-"*30)
            
            for col in feature_columns:
                while True:
                    val = input(f"Enter {col}: ").strip()
                    if val.lower() in ['exit', 'quit']:
                        should_exit = True
                        break
                    
                    try:
                        if col in encoders:
                            le = encoders[col]
                            if val not in le.classes_:
                                print(f"[INVALID] Options: {list(le.classes_)}")
                                continue
                            user_vals.append(le.transform([val])[0])
                        else:
                            user_vals.append(float(val))
                        break # Valid input, move to next feature
                    except ValueError:
                        print(f"[ERROR] {col} requires a numeric value.")
                if should_exit: break
            
            if should_exit: break
            
            # Predict
            df_input = pd.DataFrame([user_vals], columns=feature_columns)
            scaled_input = scaler.transform(df_input) # Use values to silence warnings
            
            prediction = model.predict(scaled_input)[0]
            # Decision function: The shifted local outlier factor. 
            # The more negative, the more anomalous.
            score = model.decision_function(scaled_input)[0] 
            
            if prediction == 1:
                print(f"\n**[RESULT] STATUS: NORMAL (INLIER)**")
                print(f"Local Density Score: {score:.4f} (Higher is better)")
            else:
                print(f"\n**[RESULT] STATUS: ANOMALY DETECTED (OUTLIER)**")
                print(f"Outlier Intensity: {score:.4f} (Lower/More Negative is more extreme)")
            print("-" * 30)

    return detect_loop

if __name__ == "__main__":
    # Expecting: dataset_path, n_neighbors, contamination, pkl_filename
    if len(sys.argv) == 5:
        dataset = sys.argv[1]
        neighbors = int(sys.argv[2])
        contam = float(sys.argv[3])
        filename = sys.argv[4]
        
        func = train_and_save_lof(dataset, neighbors, contam, filename)
        func()
    elif len(sys.argv) == 3 and sys.argv[2] == 'detect_only':
        func = train_and_save_lof(None, 0, 0.0, sys.argv[1])
        func()
    else:
        print("Usage: python lof_script.py <dataset_path> <n_neighbors> <contamination> <pkl_filename>")
        sys.exit(1)