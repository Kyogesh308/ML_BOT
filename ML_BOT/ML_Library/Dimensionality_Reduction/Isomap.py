import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.manifold import Isomap
# FIX 1: Removed static dataset import (MNIST)

# Added crucial imports for robust preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Constants ---
DEFAULT_FILL_VALUE = 0.0 

def train_and_save_isomap(dataset_path: str, n_components: int, n_neighbors: int, pkl_filename: str):
    """
    Trains an Isomap model (non-linear dimensionality reduction), saves the fitted model, scaler, 
    feature columns, and encoders for consistent transformation.
    """
    
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    isomap_model = None
    scaler = None
    feature_columns = None
    encoders = {}
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing Isomap model '{pkl_filename}'...**")
        try:
            # Must load all four components (model, scaler, columns, encoders)
            isomap_model, scaler, feature_columns, encoders = load(full_path) 
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
            if ext == ".csv":
                data = pd.read_csv(dataset_path)
            elif ext in [".xls", ".xlsx"]:
                data = pd.read_excel(dataset_path)
            elif ext == ".json":
                data = pd.read_json(dataset_path)
            elif ext == ".parquet":
                data = pd.read_parquet(dataset_path)
            else:
                try:
                    data = pd.read_csv(dataset_path, sep=None, engine='python')
                except:
                    with open(dataset_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    data = pd.read_csv(StringIO(content))
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}", file=sys.stderr)
            sys.exit(1)

        # --- PREPROCESSING (Template Standardization) ---

        # 1. Encode String (Categorical) Columns
        for col in data.columns:
            if data[col].dtype == object:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
        
        # 2. Impute NaNs (Numerical columns only)
        data.fillna(data.mean(numeric_only=True), inplace=True) 

        # Keep only numerical columns
        X_df = data.select_dtypes(include=[np.number])
        
        if X_df.shape[1] == 0:
            raise ValueError("Dataset contains no numeric columns after encoding.")

        # 3. Validation
        if n_components >= X_df.shape[1]:
            raise ValueError(f"n_components ({n_components}) must be less than number of features ({X_df.shape[1]}).")
        if n_neighbors >= X_df.shape[0]:
            print(f"[WARNING] n_neighbors ({n_neighbors}) is greater than number of samples ({X_df.shape[0]}). Setting n_neighbors to max available.")
            n_neighbors = X_df.shape[0] - 1
        elif n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")


        # 4. SCALING (CRITICAL for distance-based methods like Isomap)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        feature_columns = list(X_df.columns)

        # --- TRAIN ISOMAP MODEL ---
        print(f"\n**[INFO] Training Isomap model (n_components={n_components}, n_neighbors={n_neighbors})...**")

        isomap_model = Isomap(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_jobs=-1
        )

        X_reduced = isomap_model.fit_transform(X_scaled)
        print(f"[INFO] Training complete. Reduced shape: {X_reduced.shape}")

        # --- SAVE MODEL ---
        # Save all four components (model, scaler, columns, encoders)
        dump((isomap_model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] Isomap model saved as '{pkl_filename}'")
        is_trained = True

    # --- Transformation function (Unlimited Loop) ---
    def transform_loop():
        nonlocal isomap_model, scaler, feature_columns, encoders
        
        if isomap_model is None:
            try:
                # Load all four components
                isomap_model, scaler, feature_columns, encoders = load(full_path)
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model for transformation. Check the .pkl file: {e}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"ENTER FEATURE VALUES FOR DIMENSIONALITY REDUCTION (N={isomap_model.n_components} components)")
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

                        if col in encoders:
                            # Use saved encoder for categorical columns
                            le = encoders[col]
                            try:
                                # Transform the categorical value
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
                print("\n**[INFO] Transformation mode terminated by user.**")
                break

            # Convert list to DataFrame
            df_input = pd.DataFrame([user_vals], columns=feature_columns)
            
            # Scale the input using the saved scaler (resolves UserWarning)
            user_scaled = scaler.transform(df_input) 

            # Transform the scaled input using the Isomap model
            reduced = isomap_model.transform(user_scaled)
            
            print("\n" + "="*30)
            print(f"**[RESULT] Reduced Vector ({isomap_model.n_components}D):**")
            print(reduced[0]) 
            print("="*30)

    return transform_loop


if __name__ == "__main__":
    # Updated CLI checks to match the deployment standard structure
    # Now expects: dataset_path, n_components, n_neighbors, pkl_filename (4 args for train)
    
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Usage (Train/Transform): python script_name.py <dataset_path> <n_components> <n_neighbors> <pkl_filename>")
        print("Usage (Transform Only): python script_name.py <pkl_filename> transform_only")
        sys.exit(1)
        
    if len(sys.argv) == 5:
        # Standard Training and Transformation
        dataset_path = sys.argv[1]
        
        try:
            n_components = int(sys.argv[2])
            n_neighbors = int(sys.argv[3])
        except ValueError:
            print("[FATAL ERROR] n_components and n_neighbors must be integers.", file=sys.stderr)
            sys.exit(1)
            
        pkl_filename = sys.argv[4]
        
        try:
            transform_func = train_and_save_isomap(dataset_path, n_components, n_neighbors, pkl_filename)
            transform_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif len(sys.argv) == 3 and sys.argv[2].lower() == 'transform_only':
        # Transform Only Mode
        pkl_filename = sys.argv[1]
        
        try:
            # Use dummy values (0) as n_components/n_neighbors are only needed for training mode
            transform_func = train_and_save_isomap(None, 0, 0, pkl_filename) 
            transform_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)