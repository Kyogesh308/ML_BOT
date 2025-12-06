import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.manifold import TSNE
# Added crucial imports for robust preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score # Kept for template consistency, though unused here

# --- Constants ---
DEFAULT_FILL_VALUE = 0.0 

def train_and_save_tsne(dataset_path: str, n_components: int, pkl_filename: str):
    """
    Trains a t-SNE model (non-linear manifold learning for visualization), saves the fitted model, scaler, 
    feature columns, encoders, AND the reduced training set (X_embedded).
    NOTE: t-SNE does not support transforming new samples.
    """
    
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    tsne_model = None
    scaler = None
    feature_columns = None
    encoders = {}
    X_embedded = None # Store the result of the fit_transform
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing t-SNE model '{pkl_filename}'...**")
        try:
            # FIX: Must load five components: model, scaler, columns, encoders, X_embedded
            tsne_model, scaler, feature_columns, encoders, X_embedded = load(full_path) 
            print("[INFO] Model and reduced data loaded successfully.")
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

        # 4. SCALING (CRITICAL for t-SNE performance)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        feature_columns = list(X_df.columns)

        # --- TRAIN t-SNE MODEL ---
        print(f"\n**[INFO] Training t-SNE model (n_components={n_components})...**")

        tsne_model = TSNE(
            n_components=n_components,
            learning_rate="auto",
            init="random",
            perplexity=30,
            random_state=42,
            n_jobs=-1
        )

        # X_embedded stores the full transformed training set
        X_embedded = tsne_model.fit_transform(X_scaled)
        print(f"[INFO] Training complete. Reduced shape: {X_embedded.shape}")

        # --- SAVE MODEL ---
        # Save five components: model, scaler, columns, encoders, X_embedded
        dump((tsne_model, scaler, feature_columns, encoders, X_embedded), full_path)
        print(f"[INFO] t-SNE model and reduced data saved as '{pkl_filename}'")
        is_trained = True

    # --- Transformation function (Adapted for t-SNE limitation) ---
    def transform_loop():
        nonlocal X_embedded
        
        # Load X_embedded if model wasn't trained in this run
        if X_embedded is None:
            try:
                # Load X_embedded which contains the result (fifth component saved)
                _, _, _, _, X_embedded = load(full_path) 
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load reduced data for transformation. Check the .pkl file: {e}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"t-SNE WARNING: Scikit-learn's t-SNE cannot transform new samples.")
        print(f"Showing the **full reduced training dataset** (Shape: {X_embedded.shape})")
        print("="*70)
        
        # Display the pre-calculated reduced matrix.
        print("\n**[RESULT] Reduced Training Data (First 5 samples):**")
        print(X_embedded[:5]) 
        
        # Simple loop to wait for exit, maintaining template structure
        while True:
            val = input("\nType 'exit' or 'quit' to close the program: ").strip().lower()
            if val in ['exit', 'quit']:
                print("\n**[INFO] Transformation mode terminated by user.**")
                break


    return transform_loop


if __name__ == "__main__":
    # Updated CLI checks to match the deployment standard structure
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage (Train/Transform): python script_name.py <dataset_path> <n_components> <pkl_filename>")
        print("Usage (Transform Only): python script_name.py <pkl_filename> transform_only")
        sys.exit(1)
        
    if len(sys.argv) == 4:
        # Standard Training and Transformation
        dataset_path = sys.argv[1]
        pkl_filename = sys.argv[3]
        
        try:
            n_components = int(sys.argv[2])
        except ValueError:
            print("[FATAL ERROR] n_components must be an integer.", file=sys.stderr)
            sys.exit(1)
            
        try:
            transform_func = train_and_save_tsne(dataset_path, n_components, pkl_filename)
            transform_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif len(sys.argv) == 3 and sys.argv[2].lower() == 'transform_only':
        # Transform Only Mode
        pkl_filename = sys.argv[1]
        
        try:
            # Use dummy n_components (0) as it's only needed for training mode
            transform_func = train_and_save_tsne(None, 0, pkl_filename) 
            transform_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)