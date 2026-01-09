import pandas as pd
import numpy as np
import os
import sys
import warnings
from io import StringIO
from joblib import dump, load
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Silence scikit-learn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train_and_save_elliptic(dataset_path: str, contamination: float, pkl_filename: str):
    """
    Trains an Elliptic Envelope for anomaly detection.
    Assumes Gaussian (normal) distribution of data.
    """
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    model, scaler, feature_columns, encoders = None, None, None, {}
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing Elliptic Envelope model '{pkl_filename}'...**")
        try:
            model, scaler, feature_columns, encoders = load(full_path) 
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Load failed: {e}. Retraining...")

    # --- CASE 2: Train a new model ---
    if not is_trained:
        if dataset_path is None:
            print("[ERROR] No dataset provided and no saved model found.")
            return None

        # --- DATA LOADING ---
        ext = os.path.splitext(dataset_path)[1].lower()
        try:
            if ext == ".csv": data = pd.read_csv(dataset_path)
            elif ext in [".xls", ".xlsx"]: data = pd.read_excel(dataset_path)
            else: data = pd.read_csv(dataset_path, sep=None, engine='python')
        except Exception as e:
            print(f"[ERROR] Load failed: {e}")
            return None

        # --- PREPROCESSING ---
        for col in data.columns:
            if data[col].dtype == object:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
        
        data.fillna(data.mean(numeric_only=True), inplace=True) 
        X_df = data.select_dtypes(include=[np.number])
        
        # Fit scaler on DataFrame to preserve feature names for Zero-Warning
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        feature_columns = list(X_df.columns)

        # --- TRAIN ---
        print(f"\n**[INFO] Training Elliptic Envelope (contamination={contamination})...**")
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        model.fit(X_scaled)
        
        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] Model saved as '{pkl_filename}'.")

    # --- THE PREDICTION LOOP ---
    def detect_loop():
        nonlocal model, scaler, feature_columns, encoders
        
        print("\n" + "="*70 + "\nINFERENCE ENGINE: ELLIPTIC ENVELOPE\n" + "="*70)
        print("Type 'exit' or 'quit' at any time to stop.")

        while True:
            user_vals = []
            should_exit = False
            
            for col in feature_columns:
                while True: # Keep asking for THIS specific column until valid
                    val = input(f"Enter {col}: ").strip()
                    
                    if val.lower() in ['exit', 'quit']:
                        should_exit = True
                        break
                    
                    try:
                        if col in encoders:
                            le = encoders[col]
                            if val not in le.classes_:
                                print(f"[INVALID] '{val}' not recognized. Valid options: {list(le.classes_)}")
                                continue
                            user_vals.append(le.transform([val])[0])
                        else:
                            user_vals.append(float(val))
                        break # Input valid, move to next column in the feature list
                    except ValueError:
                        print(f"[ERROR] Invalid input. {col} requires a numeric value.")
                
                if should_exit: break
            
            if should_exit: break
            
            # --- PREDICTION ---
            if len(user_vals) == len(feature_columns):
                # Create DataFrame to match scaler's expected feature names
                df_input = pd.DataFrame([user_vals], columns=feature_columns)
                scaled_input = scaler.transform(df_input) # Zero-Warning: pass the DF
                
                prediction = model.predict(scaled_input)[0]
                
                print("\n" + "*"*40)
                if prediction == 1:
                    print(f"** [RESULT] STATUS: NORMAL DATA (INLIER) **")
                else:
                    print(f"** [RESULT] STATUS: ANOMALY DETECTED (OUTLIER) **")
                print("*"*40 + "\n")

    return detect_loop

if __name__ == "__main__":
    if len(sys.argv) == 4:
        # Capture the returned function and execute it
        detect_func = train_and_save_elliptic(sys.argv[1], float(sys.argv[2]), sys.argv[3])
        if detect_func:
            detect_func()
            
    elif len(sys.argv) == 3 and sys.argv[2] == 'detect_only':
        detect_func = train_and_save_elliptic(None, 0.0, sys.argv[1])
        if detect_func:
            detect_func()
    else:
        print("Usage: python script.py <dataset_path> <contamination> <pkl_filename>")