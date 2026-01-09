import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

def train_and_save_iso_forest(dataset_path: str, contamination: float, pkl_filename: str):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    model, scaler, feature_columns, encoders = None, None, None, {}
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"[DEBUG] Model file found at {full_path}. Loading...")
        try:
            model, scaler, feature_columns, encoders = load(full_path) 
            is_trained = True
            print("[INFO] Existing model loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}")

    # --- CASE 2: Train a new model ---
    if not is_trained:
        print(f"[DEBUG] Starting training phase for {dataset_path}...")
        
        # --- DATA LOADING (Robust block) ---
        try:
            ext = os.path.splitext(dataset_path)[1].lower()
            if ext == ".csv": data = pd.read_csv(dataset_path)
            elif ext in [".xls", ".xlsx"]: data = pd.read_excel(dataset_path)
            else: data = pd.read_csv(dataset_path, sep=None, engine='python')
        except Exception as e:
            print(f"[FATAL ERROR] Data loading failed: {e}")
            sys.exit(1)

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

        # --- TRAIN ---
        print(f"**[INFO] Training Isolation Forest (contamination={contamination})...**")
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_scaled)
        
        # --- SAVE ---
        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] Model saved as '{pkl_filename}'")

    # --- THE DETECTION LOOP ---
    def detect_loop():
        nonlocal model, scaler, feature_columns, encoders
        
        print("\n" + "="*70)
        print("ANOMALY DETECTION MODE: ISOLATION FOREST")
        print("Type 'exit' to quit.")
        print("="*70)

        while True:
            user_vals = []
            should_exit = False
            
            # This loop ensures we get EVERY column before trying to predict
            for col in feature_columns:
                while True: # Inner loop to keep asking for the SAME column if invalid
                    val = input(f"Enter {col}: ").strip()
                    if val.lower() in ['exit', 'quit']:
                        should_exit = True
                        break
                    
                    try:
                        if col in encoders:
                            le = encoders[col]
                            if val not in le.classes_:
                                print(f"[INVALID] Recognized options: {list(le.classes_)}")
                                continue # Ask again
                            user_vals.append(le.transform([val])[0])
                        else:
                            user_vals.append(float(val))
                        break # Success, move to the next column in feature_columns
                    except ValueError:
                        print(f"[ERROR] {col} requires a numeric value.")
                
                if should_exit: break
            
            if should_exit: break
            
            # --- CRITICAL CHECK: Only predict if we have all features ---
            if len(user_vals) == len(feature_columns):
                df_input = pd.DataFrame([user_vals], columns=feature_columns)
                scaled_input = scaler.transform(df_input.values)
                
                prediction = model.predict(scaled_input)[0]
                score = model.decision_function(scaled_input)[0]
                
                print("\n" + "*"*30)
                if prediction == 1:
                    print(f"** STATUS: NORMAL DATA (INLIER) **")
                    print(f"** Score: {score:.4f} (Positive is Normal)")
                else:
                    print(f"** STATUS: ANOMALY DETECTED (OUTLIER) **")
                    print(f"** Score: {score:.4f} (Negative is Anomalous)")
                print("*"*30 + "\n")
            else:
                print("[ERROR] Input was incomplete. Skipping prediction.")
    return detect_loop()
if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_loop = train_and_save_iso_forest(sys.argv[1], float(sys.argv[2]), sys.argv[3])
        if run_loop: # CHECK: Only call if run_loop is NOT None
            run_loop()
    elif len(sys.argv) == 3 and sys.argv[2] == 'detect_only':
        run_loop = train_and_save_iso_forest(None, 0.0, sys.argv[1])
        if run_loop:
            run_loop()
    else:
        print("Usage: python script.py <data.csv> <contamination> <model.pkl>")