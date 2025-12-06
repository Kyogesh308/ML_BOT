import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score

# --- Constants ---
DEFAULT_TEST_SIZE = 0.2

def train_and_save_model(dataset_path: str, target_column: str, pkl_filename: str):
    """
    Trains a GaussianProcessRegressor model, saves necessary components, 
    and returns a predict function.
    """
    
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    model = None
    scaler = None
    feature_columns = None
    encoders = {}
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing Gaussian Process Regressor model '{pkl_filename}'...**")
        try:
            model, scaler, feature_columns, encoders = load(full_path) 
            print("[INFO] Model loaded successfully.")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Could not load model from '{full_path}' due to error: {e}. Retraining...")
            # If load fails, we fall through to the training block
            
    # --- CASE 2: Train a new model (only if not loaded successfully) ---
    if not is_trained:
        # --- LOAD DATA ---
        ext = os.path.splitext(dataset_path)[1].lower()
        try:
            if ext == '.csv':
                data = pd.read_csv(dataset_path)
            # ... (data loading logic remains the same) ...
            elif ext in ['.xls', '.xlsx']:
                data = pd.read_excel(dataset_path)
            elif ext == '.json':
                data = pd.read_json(dataset_path)
            elif ext == '.parquet':
                data = pd.read_parquet(dataset_path)
            else:
                try:
                    data = pd.read_csv(dataset_path, sep=None, engine='python')
                except:
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    data = pd.read_csv(StringIO(content))
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}", file=sys.stderr)
            sys.exit(1)
            
        if target_column not in data.columns:
            raise KeyError(f"Target column '{target_column}' not found in the dataset.")

        y = data[target_column]
        X = data.drop(target_column, axis=1)

        # --- PREPROCESSING ---
        # 1. Encode String (Categorical) Columns
        for col in X.columns:
            if X[col].dtype == object:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
        
        # 2. Fill NA (Imputation)
        X.fillna(X.mean(numeric_only=True), inplace=True)

        # 3. SCALING
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_columns = list(X.columns)
        
        # --- SPLIT DATA ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=DEFAULT_TEST_SIZE, random_state=42
        )

        # --- TRAIN GAUSSIAN PROCESS REGRESSOR ---
        print("\n**[INFO] Training Gaussian Process Regressor...**")
        
        # NOTE ON WARNING: Increasing the upper bound of the ConstantKernel's parameter 
        # to address the ConvergenceWarning (1e3 -> 1e5)
        kernel = C(1.0, (1e-3, 1e5)) * RBF(10.0, (1e-2, 1e2))
        
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42)
        model.fit(X_train, y_train)

        # --- EVALUATION ---
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"\n[INFO] R-squared Score (Test): **{r2:.4f}**")
        print(f"[INFO] Mean Squared Error (Test): **{mse:.4f}**")

        # --- SAVE MODEL ---
        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] Model saved as '{pkl_filename}'")
        is_trained = True

    # --- Prediction function ---
    def predict_loop():
        nonlocal model, scaler, feature_columns, encoders
        
        # Reload model if training was skipped/failed
        if model is None:
            try:
                model, scaler, feature_columns, encoders = load(full_path)
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model for prediction. Training must be run first or check the .pkl file: {e}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"**Starting UNLIMITED Prediction Mode for '{pkl_filename}'**")
        print("Type 'exit' or 'quit' to close the program.")
        print("="*70)
        
        # *** FIX/FEATURE: UNLIMITED PREDICTION LOOP ***
        while True:
            user_vals = []
            
            print("\n" + "-"*30)
            
            # Use this flag to check if the loop should break
            should_exit = False
            for col in feature_columns:
                while True:
                    try:
                        val = input(f"Enter value for **{col}**: ")
                        input_val_str = str(val).strip().lower()
                        
                        if input_val_str in ['exit', 'quit']:
                            should_exit = True
                            break # Break out of inner loop
                            
                        if col in encoders:
                            le = encoders[col]
                            try:
                                encoded_val = le.transform([input_val_str])[0]
                                user_vals.append(encoded_val)
                                break
                            except ValueError:
                                known_classes = list(le.classes_)
                                if len(known_classes) > 10:
                                    known_classes = known_classes[:10] + ["..."]
                                print(f"[ERROR] Value '{input_val_str}' for column '{col}' is not a known category. Known examples: {known_classes}")
                                continue
                        else:
                            # For numeric columns
                            user_vals.append(float(val))
                            break
                    except ValueError:
                        print(f"[ERROR] Invalid number format for column '{col}'. Please enter a numeric value.")
                        continue
                
                if should_exit:
                    break # Break out of feature loop
            
            if should_exit:
                print("\n**[INFO] Prediction mode terminated by user.**")
                break # Break out of infinite while loop

            # Convert list to DataFrame and scale
            df_input = pd.DataFrame([user_vals], columns=feature_columns)
            
            # Transform without turning to numpy first to slightly clean up the warning
            # Although the warning about feature names is still possible, this is common practice.
            user_scaled = scaler.transform(df_input) 

            # GPR prediction
            pred = model.predict(user_scaled, return_std=False)[0] 
            
            print("="*30)
            print(f"**[RESULT] Predicted Continuous Outcome: {pred:.4f}**")
            print("="*30)

    return predict_loop

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Usage (Train/Predict): python script_name.py <dataset_path> <target_column> <pkl_filename>")
        print("Usage (Predict Only): python script_name.py <pkl_filename> predict_only")
        sys.exit(1)
        
    if len(sys.argv) == 4:
        # Standard Training and Prediction
        dataset_path = sys.argv[1]
        target_column = sys.argv[2]
        pkl_filename = sys.argv[3]
        
        try:
            predict_func = train_and_save_model(dataset_path, target_column, pkl_filename)
            predict_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif len(sys.argv) == 3 and sys.argv[2].lower() == 'predict_only':
        # Prediction Only Mode (Uses dummy values for unused args)
        pkl_filename = sys.argv[1]
        
        try:
            # We only need the pkl_filename to load the model
            predict_func = train_and_save_model(None, None, pkl_filename) 
            predict_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)