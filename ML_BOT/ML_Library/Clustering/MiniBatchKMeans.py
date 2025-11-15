import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load

from sklearn.preprocessing import StandardScaler, LabelEncoder # Added LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score # Added for evaluation
import warnings

# --- Constants ---
DEFAULT_N_CLUSTERS = 5 
DEFAULT_BATCH_SIZE = 256 # A typical batch size for efficiency

def train_and_save_model(dataset_path: str, pkl_filename: str):
    """
    Trains a MiniBatchKMeans model, saves necessary components, 
    and returns a predict function.
    
    The target_column argument is implicitly ignored.
    """
    
    warnings.filterwarnings("ignore", category=UserWarning)

    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    model_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    model = None
    scaler = None
    feature_columns = None
    encoders = {}

    ## --- CASE 1: Load existing model ---
    if os.path.exists(model_path):
        print(f"**[INFO] Loading existing MiniBatchKMeans model '{pkl_filename}'...**")
        # Ensure four components are loaded
        model, scaler, feature_columns, encoders = load(model_path)
        print("[INFO] Model loaded successfully.")

    ## --- CASE 2: Train a new model ---
    else:
        # --- LOAD DATA (Robust Multi-File Loading) ---
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

        # X is the entire dataset (unsupervised, target_column is ignored)
        X = data.copy()

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

        # --- TRAIN MiniBatchKMeans ---
        print(f"\n**[INFO] Training MiniBatchKMeans with n_clusters={DEFAULT_N_CLUSTERS}...**")
        model = MiniBatchKMeans(
            n_clusters=DEFAULT_N_CLUSTERS,
            batch_size=DEFAULT_BATCH_SIZE,
            random_state=42,
            n_init='auto'
        )
        labels = model.fit_predict(X_scaled)
        print("[INFO] Training complete.")

        # --- EVALUATION ---
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            print(f"[INFO] Silhouette Score (n_clusters={DEFAULT_N_CLUSTERS}): **{score:.4f}**")
        else:
             print("[INFO] Only 1 effective cluster found. Silhouette Score not calculated.")

        # --- SAVE MODEL ---
        # Save the model, scaler, feature names, and encoders
        dump((model, scaler, feature_columns, encoders), model_path)
        print(f"[INFO] Model saved as '{pkl_filename}'")

    # --- Prediction function ---
    def predict():
        nonlocal model, scaler, feature_columns, encoders
        
        # Load components if not already in memory
        if model is None:
            try:
                model, scaler, feature_columns, encoders = load(model_path)
            except Exception as e:
                print(f"[ERROR] Failed to load model for prediction: {e}", file=sys.stderr)
                return

        print("\n" + "="*50)
        print("ENTER FEATURE VALUES FOR CLUSTER PREDICTION (MINIBATCHKMEANS)")
        print("="*50)
        user_vals = []
        
        for col in feature_columns:
            while True:
                try:
                    val = input(f"Enter value for **{col}**: ")
                    
                    # Apply encoding if needed
                    if col in encoders:
                        le = encoders[col]
                        try:
                            encoded_val = le.transform([str(val).strip()])[0]
                            user_vals.append(encoded_val)
                            break
                        except ValueError:
                            print(f"[ERROR] Value '{val}' for column '{col}' is not a known category. Known: {list(le.classes_)}")
                            continue
                    else:
                        # Handle numerical input
                        user_vals.append(float(val))
                        break
                except ValueError:
                    print(f"[ERROR] Invalid number format for column '{col}'. Please enter a numeric value.")
                    continue

        # Convert list to DataFrame and scale
        df_input = pd.DataFrame([user_vals], columns=feature_columns)
        # FIX: Use .to_numpy() to avoid feature name warning
        user_scaled = scaler.transform(df_input.to_numpy())

        # Predict cluster using MiniBatchKMeans's native .predict()
        cluster_id = model.predict(user_scaled)[0]
        
        print("\n" + "="*50)
        print(f"**[RESULT] Assigned Cluster: {cluster_id}**")
        print("="*50)

    return predict


if __name__ == "__main__":
    if len(sys.argv) != 3:
        # Corrected argument count for unsupervised learning
        print("Usage: python script_name.py <dataset_path> <pkl_filename>")
        print("Note: The target_column argument is implicitly ignored.")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    pkl_filename = sys.argv[2]
    
    try:
        # NOTE: target_column is not passed to the main function
        predict_func = train_and_save_model(dataset_path, pkl_filename)
        predict_func()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)