import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load

from sklearn.cluster import OPTICS
from sklearn.neighbors import KNeighborsClassifier # Used for prediction proxy
from sklearn.preprocessing import StandardScaler, LabelEncoder # Added LabelEncoder
from sklearn.metrics import silhouette_score # Added for evaluation
import warnings

# --- Constants ---
DEFAULT_MIN_SAMPLES = 10 # Minimum number of samples in a neighborhood
DEFAULT_MIN_CLUSTER_SIZE = 5 # Minimum size of a cluster (OPTICS parameter)

def train_and_save_model(dataset_path: str, pkl_filename: str):
    """
    Trains an OPTICS Clustering model, uses a KNN proxy for prediction, 
    saves necessary components, and returns a predict function.
    
    The target_column argument is implicitly ignored.
    """
    
    # Suppress warnings related to OPTICS/KNN/scaling
    warnings.filterwarnings("ignore", category=UserWarning)

    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    model_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    model = None          # KNeighborsClassifier proxy
    scaler = None
    feature_columns = None
    encoders = {}

    ## --- CASE 1: Load existing model ---
    if os.path.exists(model_path):
        print(f"**[INFO] Loading existing OPTICS predictive model '{pkl_filename}'...**")
        # Ensure four components are loaded
        model, scaler, feature_columns, encoders = load(model_path)
        print("[INFO] Predictive model loaded successfully.")

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

        # --- TRAIN OPTICS MODEL ---
        print(f"\n**[INFO] Training OPTICS Clustering...**")
        
        optics_model = OPTICS(
            min_samples=DEFAULT_MIN_SAMPLES, 
            min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE
        )
        optics_model.fit(X_scaled)

        # OPTICS requires extraction to get labels. We use the most basic method:
        labels = optics_model.labels_

        # --- CREATE PREDICTION PROXY (MANDATORY for OPTICS inference) ---
        # OPTICS has no .predict(). We use KNeighborsClassifier trained on the scaled data and the resulting OPTICS labels.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 0:
            print("[INFO] Creating KNeighborsClassifier proxy for prediction...")
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_scaled, labels)
        else:
            print("[WARNING] OPTICS found 0 clusters (all data is noise). Using basic KNN proxy.")
            model = KNeighborsClassifier(n_neighbors=1) 
            model.fit(X_scaled, labels)


        # --- EVALUATION ---
        # Filter out noise points (-1) for a valid silhouette score.
        mask = (labels != -1)
        if np.sum(mask) > 1 and n_clusters > 1:
            score = silhouette_score(X_scaled[mask], labels[mask])
            print(f"\n[INFO] Silhouette Score (excluding noise): **{score:.4f}**")
        else:
             print("[INFO] Silhouette Score not calculated (too few points or clusters).")

        # --- SAVE MODEL ---
        # Save the KNN proxy, scaler, feature names, and encoders
        dump((model, scaler, feature_columns, encoders), model_path)
        print(f"[INFO] Predictive model (KNeighborsClassifier) saved as '{pkl_filename}'")
    
    
    # --- Prediction function ---
    def predict():
        nonlocal model, scaler, feature_columns, encoders
        
        if model is None:
            try:
                model, scaler, feature_columns, encoders = load(model_path)
            except Exception as e:
                print(f"[ERROR] Failed to load model for prediction: {e}", file=sys.stderr)
                return

        print("\n" + "="*50)
        print("ENTER FEATURE VALUES FOR CLUSTER PREDICTION (OPTICS/KNN)")
        print("="*50)
        user_vals = []
        
        for col in feature_columns:
            while True:
                try:
                    val = input(f"Enter value for **{col}**: ")
                    
                    if col in encoders:
                        le = encoders[col]
                        try:
                            encoded_val = le.transform([str(val).strip()])[0]
                            user_vals.append(encoded_val)
                            break
                        except ValueError:
                            print(f"[ERROR] Value '{val}' for column '{col}' is not a known category. Please re-enter.")
                            continue
                    else:
                        user_vals.append(float(val))
                        break
                except ValueError:
                    print(f"[ERROR] Invalid number format for column '{col}'. Please enter a numeric value.")
                    continue

        user_df = pd.DataFrame([user_vals], columns=feature_columns)
        # Use .to_numpy() to avoid feature name warning
        user_scaled = scaler.transform(user_df.to_numpy())

        # Predict cluster using the KNN proxy model
        cluster = model.predict(user_scaled)[0]
        
        cluster_result = f"Cluster: {cluster}"
        if cluster == -1:
            cluster_result = "Noise/Outlier (-1)"

        print("\n" + "="*50)
        print(f"**[RESULT] Assigned {cluster_result}**")
        print("="*50)

    return predict


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <dataset_path> <pkl_filename>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    pkl_filename = sys.argv[2]
    
    try:
        predict_func = train_and_save_model(dataset_path, pkl_filename)
        predict_func()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)