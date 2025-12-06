import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.model_selection import train_test_split
# Added crucial imports for classification standard
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score 

# --- Constants ---
DEFAULT_TEST_SIZE = 0.2

def train_and_save_model(dataset_path: str, target_column: str, pkl_filename: str):
    """
    Trains a GaussianNB model, saves all five necessary components 
    (model, scaler, feature encoders, target encoder), and returns an unlimited predict function.
    """
    
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading (5 Components for Classification)
    model = None
    scaler = None
    feature_columns = None
    feature_encoders = {}
    target_encoder = None
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing Gaussian Naive Bayes model '{pkl_filename}'...**")
        try:
            # Load all five required components
            model, scaler, feature_columns, feature_encoders, target_encoder = load(full_path) 
            print("[INFO] Model loaded successfully.")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Could not load model from '{full_path}' due to error: {e}. Retraining...")
            pass
            
    # --- CASE 2: Train a new model (or retrain if loading failed) ---
    if not is_trained:
        # --- LOAD DATA ---
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
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    data = pd.read_csv(StringIO(content))
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}", file=sys.stderr)
            sys.exit(1)
            
        if target_column not in data.columns:
            raise KeyError(f"Target column '{target_column}' not found in the dataset.")

        y_raw = data[target_column].astype(str)
        X = data.drop(target_column, axis=1)

        # --- PREPROCESSING ---

        # 1. Encode Target Variable (y) (CRITICAL)
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw)

        # 2. Encode Feature Columns (X)
        for col in X.columns:
            if X[col].dtype == object:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                feature_encoders[col] = le
        
        # 3. Fill NA (Imputation)
        X.fillna(X.mean(numeric_only=True), inplace=True)

        # 4. SCALING (Recommended for Naive Bayes)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_columns = list(X.columns)
        
        # --- SPLIT DATA ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=DEFAULT_TEST_SIZE, random_state=42
        )

        # --- TRAIN GAUSSIAN NB CLASSIFIER ---
        print("\n**[INFO] Training Gaussian Naive Bayes Classifier...**")
        
        model = GaussianNB() 
        model.fit(X_train, y_train)

        # --- EVALUATION ---
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n[INFO] Accuracy Score (Test): **{acc:.4f}**")
        print(f"[INFO] F1 Score (Weighted Test): **{f1:.4f}**")

        # --- SAVE MODEL ---
        # Save all five components: model, scaler, feature columns, feature encoders, target encoder
        dump((model, scaler, feature_columns, feature_encoders, target_encoder), full_path)
        print(f"[INFO] Model saved as '{pkl_filename}'")
        is_trained = True

    # --- Prediction function (Unlimited Loop) ---
    def predict_loop():
        nonlocal model, scaler, feature_columns, feature_encoders, target_encoder
        
        if model is None:
            try:
                # Load all five components
                model, scaler, feature_columns, feature_encoders, target_encoder = load(full_path)
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model for prediction: {e}", file=sys.stderr)
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
                        input_val_str = str(val).strip()

                        if input_val_str.lower() in ['exit', 'quit']:
                            should_exit = True
                            break
                            
                        if col in feature_encoders:
                            le = feature_encoders[col]
                            try:
                                # Transform categorical input
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
            
            # Apply scaling (Pass DataFrame directly to scaler to resolve UserWarning)
            user_scaled = scaler.transform(df_input) 
            
            # Predict encoded label
            pred_encoded = model.predict(user_scaled)[0] 
            
            # Decode the prediction back to the original label
            pred_label = target_encoder.inverse_transform([pred_encoded])[0]
            
            print("="*30)
            print(f"**[RESULT] Predicted Class: {pred_label}**")
            print("="*30)

    return predict_loop

if __name__ == "__main__":
    # Updated CLI checks to match the deployment standard structure
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
            # Use dummy None values for path/target column
            predict_func = train_and_save_model(None, None, pkl_filename) 
            predict_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)