import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.decomposition import LatentDirichletAllocation
# Added crucial imports for robust preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Constants ---
DEFAULT_FILL_VALUE = 0.0 

def train_and_save_lda(dataset_path: str, n_components: int, pkl_filename: str):
    """
    Trains a LatentDirichletAllocation (LDA) model (for non-negative integer count data), 
    and saves the fitted model, feature columns, and encoders for consistent transformation.
    """
    
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    lda_model = None
    scaler = None # Initialized for template consistency
    feature_columns = None
    encoders = {}
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing LDA model '{pkl_filename}'...**")
        try:
            # FIX 3: Must load all four components (model, scaler, columns, encoders)
            lda_model, scaler, feature_columns, encoders = load(full_path) 
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

        # 1. Encode String (Categorical) Columns (FIX 2)
        for col in data.columns:
            if data[col].dtype == object:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le
        
        # 2. Impute NaNs (Numerical columns only)
        data.fillna(data.mean(numeric_only=True), inplace=True) 

        # Keep only numerical columns for LDA
        X = data.select_dtypes(include=[np.number])

        # 3. LDA-Specific Data Validation and Preparation
        if (X < 0).any().any():
            raise ValueError("LDA requires non-negative count data. Found negative values after imputation/encoding.")

        # LDA should typically work with integers, rounding and casting is kept.
        X = X.round().astype(int)

        if X.shape[1] == 0:
            raise ValueError("Dataset contains no numeric columns after encoding.")

        # 4. Validation
        if n_components >= X.shape[1]:
            print(f"[WARNING] n_components ({n_components}) is not strictly less than the number of features ({X.shape[1]}). LDA will run, but this is usually sub-optimal.")
        
        # 5. SCALING (NOT applied to the data for LDA, but scaler object must be created/saved for template consistency)
        scaler = StandardScaler()
        # NOTE: Scaler is NOT fitted or applied to X, as LDA works on raw counts.
        feature_columns = list(X.columns)

        # --- TRAIN LDA MODEL ---
        print(f"\n**[INFO] Training LatentDirichletAllocation model (n_components={n_components})...**")

        lda_model = LatentDirichletAllocation(
            n_components=n_components,
            learning_method="batch",
            max_iter=50,
            random_state=42
        )

        X_reduced = lda_model.fit_transform(X) # X is raw count data
        print(f"[INFO] Training complete. Reduced shape: {X_reduced.shape}")

        # --- SAVE MODEL ---
        # FIX 3: Save all four components (model, scaler, columns, encoders)
        dump((lda_model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] LDA model saved as '{pkl_filename}'")
        is_trained = True

    # --- Transformation function (Unlimited Loop) ---
    def transform_loop():
        nonlocal lda_model, scaler, feature_columns, encoders
        
        if lda_model is None:
            try:
                # Load all four components
                lda_model, scaler, feature_columns, encoders = load(full_path)
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model for transformation. Check the .pkl file: {e}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"ENTER FEATURE VALUES FOR DIMENSIONALITY REDUCTION (N={lda_model.n_components} topics)")
        print("NOTE: Input must be non-negative integers (counts).")
        print("Type 'exit' or 'quit' to close the program.")
        print("="*70)

        # FIX 5: Converted to unlimited transformation loop
        while True:
            user_vals = []
            print("\n" + "-"*30)
            
            should_exit = False
            for col in feature_columns:
                while True:
                    try:
                        val = input(f"Enter value for **{col}** (count): ")
                        input_val_str = str(val).strip()

                        if input_val_str.lower() in ['exit', 'quit']:
                            should_exit = True
                            break
                        
                        # Validate for non-negative integer counts
                        count_val = int(input_val_str)
                        if count_val < 0:
                             raise ValueError("Input must be non-negative.")
                        
                        if col in encoders:
                            # Use saved encoder for categorical columns if needed
                            le = encoders[col]
                            # For LDA, the *input* is often raw features before encoding, 
                            # but if categorical features were label-encoded *before* being used
                            # as features, we must apply the encoding here.
                            if X.columns.get_loc(col) < len(encoders): # Check if column was originally categorical
                                try:
                                    encoded_val = le.transform([input_val_str])[0]
                                    user_vals.append(encoded_val)
                                    break
                                except ValueError:
                                    print(f"[ERROR] Value '{input_val_str}' for column '{col}' is not a known category.")
                                    continue
                        
                        # For numeric/count columns
                        user_vals.append(count_val)
                        break

                    except ValueError as e:
                        print(f"[ERROR] Invalid number format or constraint violated for column '{col}'. Must be a non-negative integer count. Error: {e}")
                        continue
                
                if should_exit:
                    break
            
            if should_exit:
                print("\n**[INFO] Transformation mode terminated by user.**")
                break

            # Convert list to DataFrame
            df_input = pd.DataFrame([user_vals], columns=feature_columns)
            
            # SCALING (NOT APPLIED to data, but necessary if template was strict)
            # Since LDA requires non-negative counts, we skip scaling the actual data.
            user_scaled = df_input.to_numpy() # Direct conversion to matrix

            # Transform the raw input using the LDA model
            reduced = lda_model.transform(user_scaled)
            
            print("\n" + "="*30)
            print(f"**[RESULT] Reduced Topic Distribution Vector ({lda_model.n_components}D):**")
            print(reduced[0]) 
            print("="*30)

    return transform_loop


if __name__ == "__main__":
    # FIX 4: Updated CLI checks to match the deployment standard structure
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage (Train/Transform): python script_name.py <dataset_path> <n_components> <pkl_filename>")
        print("Usage (Transform Only): python script_name.py <pkl_filename> transform_only")
        sys.exit(1)
        
    if len(sys.argv) == 4:
        # Standard Training and Transformation
        dataset_path = sys.argv[1]
        
        try:
            n_components = int(sys.argv[2])
        except ValueError:
            print("[FATAL ERROR] n_components must be an integer.", file=sys.stderr)
            sys.exit(1)
            
        pkl_filename = sys.argv[3]
        
        try:
            transform_func = train_and_save_lda(dataset_path, n_components, pkl_filename)
            transform_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif len(sys.argv) == 3 and sys.argv[2].lower() == 'transform_only':
        # Transform Only Mode
        pkl_filename = sys.argv[1]
        
        try:
            # Use dummy n_components (0) as it's only needed for training mode
            transform_func = train_and_save_lda(None, 0, pkl_filename) 
            transform_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)