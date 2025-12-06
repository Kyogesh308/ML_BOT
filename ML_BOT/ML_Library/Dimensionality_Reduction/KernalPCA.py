import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
# Added crucial import for robust preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA

# --- Constants ---
DEFAULT_FILL_VALUE = 0.0 
DEFAULT_KERNEL = "rbf"

def train_and_save_reducer(dataset_path: str, n_components: int, pkl_filename: str, kernel: str = DEFAULT_KERNEL):
    """
    Trains a KernelPCA model (non-linear dimensionality reduction), saves the fitted model, scaler, 
    feature columns, and encoders for consistent transformation.
    """
    
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    reducer = None
    scaler = None
    feature_columns = None
    encoders = {}
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing KernelPCA model (Kernel: {kernel}) '{pkl_filename}'...**")
        try:
            # FIX 3: Must load all four components (model, scaler, columns, encoders)
            reducer, scaler, feature_columns, encoders = load(full_path) 
            print("[INFO] Reducer loaded successfully.")
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

        # 1. Encode String (Categorical) Columns (FIX 2)
        for col in data.columns:
            if data[col].dtype == object:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
        
        # 2. Impute NaNs (Numerical columns only)
        data.fillna(data.mean(numeric_only=True), inplace=True) 

        # Keep only numerical columns
        X = data.select_dtypes(include=[np.number])
        
        if X.shape[1] == 0:
            raise ValueError("Dataset contains no numeric columns after encoding.")

        # 3. Validation
        if n_components >= X.shape[1]:
            raise ValueError(f"n_components ({n_components}) must be less than number of features ({X.shape[1]}).")

        # 4. SCALING (CRITICAL for PCA variants)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_columns = list(X.columns)

        # --- TRAIN KernelPCA MODEL ---
        print(f"\n**[INFO] Training KernelPCA model (n_components={n_components}, kernel='{kernel}')...**")

        reducer = KernelPCA(
            n_components=n_components, 
            kernel=kernel, 
            fit_inverse_transform=True,
            n_jobs=-1
        )
        X_reduced = reducer.fit_transform(X_scaled)
        print(f"[INFO] Training complete. Reduced shape: {X_reduced.shape}")

        # --- SAVE MODEL ---
        # FIX 3: Save all four components (reducer, scaler, columns, encoders)
        dump((reducer, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] KernelPCA model saved as '{pkl_filename}'")
        is_trained = True

    # --- Transformation function (Unlimited Loop) ---
    def transform_loop():
        nonlocal reducer, scaler, feature_columns, encoders
        
        if reducer is None:
            try:
                # Load all four components
                reducer, scaler, feature_columns, encoders = load(full_path)
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model for transformation. Check the .pkl file: {e}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"ENTER FEATURE VALUES FOR DIMENSIONALITY REDUCTION (N={reducer.n_components} components, Kernel='{reducer.kernel}')")
        print("Type 'exit' or 'quit' to close the program.")
        print("="*70)

        # FIX 5: Converted to unlimited prediction loop
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
            
            # FIX: Scale the input using the saved scaler (resolves UserWarning)
            user_scaled = scaler.transform(df_input) 

            # Transform the scaled input using the reducer model
            reduced = reducer.transform(user_scaled)
            
            print("\n" + "="*30)
            print(f"**[RESULT] Reduced Vector ({reducer.n_components}D):**")
            print(reduced[0]) 
            print("="*30)

    return transform_loop


if __name__ == "__main__":
    # FIX 4: Updated CLI checks to handle optional kernel argument and transform_only mode
    
    # Arg count: 4 (path, n_comp, pkl_name, kernel) or 3 (path, n_comp, pkl_name) or 3 (pkl_name, transform_only)
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage (Train/Transform): python script_name.py <dataset_path> <n_components> <pkl_filename> [kernel]")
        print("Usage (Transform Only): python script_name.py <pkl_filename> transform_only")
        sys.exit(1)
        
    if len(sys.argv) >= 4 and sys.argv[2].lower() != 'transform_only':
        # Standard Training and Transformation
        dataset_path = sys.argv[1]
        pkl_filename = sys.argv[3]
        kernel = sys.argv[4] if len(sys.argv) == 5 else DEFAULT_KERNEL
        
        try:
            n_components = int(sys.argv[2])
        except ValueError:
            print("[FATAL ERROR] n_components must be an integer.", file=sys.stderr)
            sys.exit(1)
            
        try:
            transform_func = train_and_save_reducer(dataset_path, n_components, pkl_filename, kernel)
            transform_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif len(sys.argv) == 3 and sys.argv[2].lower() == 'transform_only':
        # Transform Only Mode
        pkl_filename = sys.argv[1]
        
        try:
            # Use dummy values (0, default kernel) as n_components/kernel are only needed for training mode
            transform_func = train_and_save_reducer(None, 0, pkl_filename, DEFAULT_KERNEL) 
            transform_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid combination of arguments provided.")
        sys.exit(1)