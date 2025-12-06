#this code has errors! 
import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, f1_score 

# --- Constants ---
DEFAULT_TEST_SIZE = 0.2
# Placeholder for missing values, guaranteed to be encoded as one specific index.
MISSING_CAT_LABEL = 'MISSING_VALUE_LABEL' 

def train_and_save_model(dataset_path: str, target_column: str, pkl_filename: str):
    """
    Trains a CategoricalNB model (for discrete categorical data), saves all five necessary components 
    (model, scaler, feature encoders, target encoder), and returns an unlimited predict function.
    NOTE: Data is encoded but NOT SCALED, as CategoricalNB requires integer counts/indices.
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
        print(f"**[INFO] Loading existing Categorical Naive Bayes model '{pkl_filename}'...**")
        try:
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

        # 2. Encode ALL Feature Columns (X) to non-negative integers (MANDATORY for CategoricalNB)
        X_encoded = X.copy()
        
        for col in X.columns:
            le = LabelEncoder()
            
            # IMPUTATION BEFORE ENCODING: Treat NaNs as a dedicated category
            col_data = X[col].astype(str).fillna(MISSING_CAT_LABEL) 
            
            # Fit/Transform the data, including the 'MISSING_VALUE_LABEL' category
            X_encoded[col] = le.fit_transform(col_data)
            feature_encoders[col] = le
            
        # 3. Final Imputation Check (Should be clean, but for safety)
        X_encoded.fillna(0, inplace=True) 
        
        # Ensure all columns are non-negative integers as required by CategoricalNB
        X_train_data = X_encoded.round().astype(int)
        
        # 4. SCALING (SKIPPED for CategoricalNB, but scaler object must be created and saved)
        scaler = StandardScaler()
        # X_scaled is X_train_data for CategoricalNB
        X_scaled = X_train_data 
        feature_columns = list(X_encoded.columns)
        
        # --- SPLIT DATA ---
        # Splitting the integer-encoded, NON-SCALED data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=DEFAULT_TEST_SIZE, random_state=42
        )

        # --- TRAIN CATEGORICAL NB CLASSIFIER ---
        print("\n**[INFO] Training Categorical Naive Bayes Classifier...**")
        
        model = CategoricalNB() 
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
                        
                        # Data processing for CategoricalNB: Input must be encoded but NOT scaled.
                        
                        if col in feature_encoders:
                            # Use saved encoder for categorical columns
                            le = feature_encoders[col]
                            try:
                                # Transform categorical input using the saved encoder
                                encoded_val = le.transform([input_val_str])[0]
                                user_vals.append(encoded_val)
                                break
                            except ValueError:
                                # This handles unseen categories
                                known_classes = list(le.classes_)
                                if len(known_classes) > 10:
                                    known_classes = known_classes[:10] + ["..."]
                                print(f"[ERROR] Value '{input_val_str}' for column '{col}' is not a known category. Known examples: {known_classes}")
                                continue
                        else:
                            # If feature did not require encoding (rare, but possible if it was clean integer data)
                            # We must ensure the input is cast to integer for the model.
                            try:
                                user_vals.append(int(float(input_val_str))) 
                                break
                            except ValueError:
                                print(f"[ERROR] Invalid value for column '{col}'. Expected an integer/numeric value.")
                                continue
                            
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
            
            # The data must NOT be scaled for CategoricalNB
            user_input_for_model = df_input.to_numpy()
            
            # Predict encoded label
            pred_encoded = model.predict(user_input_for_model)[0] 
            
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