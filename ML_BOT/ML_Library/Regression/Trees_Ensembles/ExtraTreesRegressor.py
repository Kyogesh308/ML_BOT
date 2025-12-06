import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.model_selection import train_test_split
# Added missing imports from template
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Constants ---
DEFAULT_TEST_SIZE = 0.2
# Extra Trees parameters
DEFAULT_N_ESTIMATORS = 100 
DEFAULT_MAX_DEPTH = 10

def train_and_save_model(dataset_path: str, target_column: str, pkl_filename: str):
    """
    Trains an ExtraTreesRegressor model (Extremely Randomized Trees), saves necessary components, 
    and returns an unlimited predict function.
    """
    
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True) # Ensure directory exists
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    model = None
    scaler = None
    feature_columns = None
    encoders = {} # Initialize encoders for template consistency
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing Extra Trees Regressor (n_estimators={DEFAULT_N_ESTIMATORS}) model '{pkl_filename}'...**")
        try:
            # Load all four required components
            model, scaler, feature_columns, encoders = load(full_path) 
            print("[INFO] Model loaded successfully.")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Could not load model from '{full_path}' due to error: {e}. Retraining...")
            pass
            
    # --- CASE 2: Train a new model (or retrain if loading failed) ---
    if not is_trained:
        # --- LOAD DATA (Robust multi-file loading) ---
        ext = os.path.splitext(dataset_path)[1].lower()
        try:
            if ext == '.csv':
                data = pd.read_csv(dataset_path)
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
                    # Fallback for generic text files
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

        # 3. SCALING (Optional but kept for consistency)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_columns = list(X.columns)
        
        # --- SPLIT DATA ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=DEFAULT_TEST_SIZE, random_state=42
        )

        # --- TRAIN EXTRA TREES REGRESSOR ---
        print(f"\n**[INFO] Training Extra Trees Regressor (n_estimators={DEFAULT_N_ESTIMATORS})...**")
        
        # ExtraTreesRegressor initialization
        model = ExtraTreesRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS, 
            max_depth=DEFAULT_MAX_DEPTH, 
            random_state=42, 
            n_jobs=-1
        ) 
        model.fit(X_train, y_train)

        # --- EVALUATION ---
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"\n[INFO] R-squared Score (Test): **{r2:.4f}**")
        print(f"[INFO] Mean Squared Error (Test): **{mse:.4f}**")

        # --- SAVE MODEL ---
        # Save all four components
        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] Model saved as '{pkl_filename}'")
        is_trained = True

    # --- Prediction function (Unlimited Loop) ---
    def predict_loop():
        nonlocal model, scaler, feature_columns, encoders
        
        if model is None:
            try:
                model, scaler, feature_columns, encoders = load(full_path)
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model for prediction. Check the .pkl file: {e}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"**Starting UNLIMITED Prediction Mode for '{pkl_filename}'**")
        print("Type 'exit' or 'quit' to close the program.")
        print("="*70)
        
        while True:
            user_vals = []
            print("\n" + "-"*30)
            
            should_exit = False
            for col in feature_columns:
                while True:
                    try:
                        val = input(f"Enter value for **{col}**: ")
                        input_val_str = str(val).strip().lower()
                        
                        if input_val_str in ['exit', 'quit']:
                            should_exit = True
                            break
                            
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
                    break
            
            if should_exit:
                print("\n**[INFO] Prediction mode terminated by user.**")
                break

            # Convert list to DataFrame
            df_input = pd.DataFrame([user_vals], columns=feature_columns)
            
            # FIX: Pass DataFrame directly to scaler to resolve UserWarning
            user_scaled = scaler.transform(df_input) 
            
            pred = model.predict(user_scaled)[0] 
            
            print("="*30)
            print(f"**[RESULT] Predicted Continuous Outcome: {pred:.4f}**")
            print("="*30)

    return predict_loop

if __name__ == "__main__":
    # Updated CLI to handle Prediction Only mode
    if len(sys.argv) < 2 or len(sys.argv) > 4:
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
        # Prediction Only Mode
        pkl_filename = sys.argv[1]
        
        try:
            # Pass None for data path/target column as they are not needed for prediction
            predict_func = train_and_save_model(None, None, pkl_filename) 
            predict_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)