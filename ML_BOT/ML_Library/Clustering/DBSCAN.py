import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier # Used for prediction proxy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import warnings

# --- Constants ---
DEFAULT_EPS = 0.5    # The maximum distance
DEFAULT_MIN_SAMPLES = 5 
DEFAULT_K_NEIGHBORS = 5 # K value for the proxy classifier

def train_and_save_dbscan(dataset_path: str, pkl_filename: str):
    """
    Trains a DBSCAN model, uses a KNeighborsClassifier as a proxy for prediction, 
    saves necessary components, and returns a predict function.
    NOTE: DBSCAN is density-based and does not require an input for n_clusters (K).
    """
    
    # Suppress warnings related to convergence and initialization
    warnings.filterwarnings("ignore", category=UserWarning)

    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables to store components for saving/loading
    model = None          # Stores the KNeighborsClassifier proxy (or DBSCAN if no clusters found)
    scaler = None
    feature_columns = None
    encoders = {}
    is_trained = False
    
    # DBSCAN parameters are fixed/defaulted, so they don't need to be loaded as a separate component.

    ## --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing DBSCAN predictive model '{pkl_filename}'...**")
        try:
            # Load the four required components
            model, scaler, feature_columns, encoders = load(full_path)
            print("[INFO] Predictive model loaded successfully.")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Could not load model from '{full_path}' due to error: {e}. Retraining...")
            pass

    ## --- CASE 2: Train and Save a new model ---
    if not is_trained:
        ## --- LOAD DATA ---
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
                    with open(dataset_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    data = pd.read_csv(StringIO(content))
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}", file=sys.stderr)
            sys.exit(1)

        # X is the entire dataset (unsupervised)
        X = data.copy()

        ## --- PREPROCESSING ---

        # 1. Encode String (Categorical) Columns
        for col in X.columns:
            if X[col].dtype == object:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

        # 2. Fill NA (Imputation)
        X.fillna(X.mean(numeric_only=True), inplace=True)

        # 3. SCALING (CRITICAL for distance-based clustering)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        feature_columns = list(X.columns)

        ## --- TRAIN DBSCAN MODEL ---
        print(f"\n**[INFO] Training DBSCAN Clustering with eps={DEFAULT_EPS} and min_samples={DEFAULT_MIN_SAMPLES}...**")
        
        dbscan_model = DBSCAN(
            eps=DEFAULT_EPS, 
            min_samples=DEFAULT_MIN_SAMPLES,
            n_jobs=-1
        )

        dbscan_model.fit(X_scaled)
        labels = dbscan_model.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[INFO] Estimated number of clusters found: {n_clusters}")

        ## --- CREATE PREDICTION PROXY (MANDATORY for DBSCAN inference) ---
        if n_clusters > 0:
            # We train the KNN proxy only on points that are NOT noise (-1)
            X_clustered = X_scaled[labels != -1]
            labels_clustered = labels[labels != -1]
            
            if len(labels_clustered) > 0:
                 print(f"[INFO] Creating KNeighborsClassifier proxy (K={DEFAULT_K_NEIGHBORS}) using {len(X_clustered)} clustered points...")
                 model = KNeighborsClassifier(n_neighbors=DEFAULT_K_NEIGHBORS, n_jobs=-1)
                 model.fit(X_clustered, labels_clustered)
            else:
                 print("[WARNING] DBSCAN found 0 clusters. Falling back to predicting noise (-1).")
                 # Setting model to DBSCAN instance itself to signify that only noise will be predicted
                 model = dbscan_model 
        else:
             print("[WARNING] DBSCAN found 0 clusters (all data is noise). Setting model to DBSCAN instance.")
             model = dbscan_model

        ## --- EVALUATION ---
        mask = (labels != -1)
        if np.sum(mask) > 1 and n_clusters > 1:
            score = silhouette_score(X_scaled[mask], labels[mask])
            print(f"\n[INFO] Silhouette Score (excluding noise): **{score:.4f}**")
        else:
            print("[INFO] Silhouette Score not calculated (too few non-noise points or clusters).")

        ## --- SAVE MODEL COMPONENTS ---
        # Save the predictive model (KNeighbors/DBSCAN instance), scaler, feature names, and encoders
        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"\n[INFO] Predictive model saved as '{pkl_filename}'")
        is_trained = True
    
    
    ## --- PREDICT FUNCTION (Unlimited Loop for new data inference) ---
    def predict_loop():
        nonlocal model, scaler, feature_columns, encoders
        
        if model is None:
            try:
                # Load the four required components
                model, scaler, feature_columns, encoders = load(full_path)
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model for prediction: {e}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"**Starting UNLIMITED Cluster Prediction Mode for '{pkl_filename}'**")
        print("DBSCAN assigns Noise points to -1.")
        print("Type 'exit' or 'quit' to close the program.")
        print("="*70)

        while True:
            user_vals = []
            print("\n" + "-"*30)
            
            should_exit = False
            for col in feature_columns:
                while True:
                    try:
                        value = input(f"Enter value for **{col}**: ")
                        input_val_str = str(value).strip()

                        if input_val_str.lower() in ['exit', 'quit']:
                            should_exit = True
                            break
                            
                        # Handle categorical columns using the saved encoders
                        if col in encoders:
                            le = encoders[col]
                            try:
                                encoded_value = le.transform([input_val_str])[0]
                                user_vals.append(encoded_value)
                                break
                            except ValueError:
                                known_classes = list(le.classes_)
                                if len(known_classes) > 10:
                                    known_classes = known_classes[:10] + ["..."]
                                print(f"[ERROR] Value '{input_val_str}' for column '{col}' is not a known category. Known examples: {known_classes}")
                                continue
                        
                        # Handle numerical columns
                        else:
                            user_vals.append(float(value))
                            break
                    except ValueError:
                        print(f"[ERROR] Invalid number format for column '{col}'. Please enter a numeric value.")
                        continue
                
                if should_exit:
                    break
            
            if should_exit:
                print("\n**[INFO] Prediction mode terminated by user.**")
                break

            # Convert user input to a DataFrame
            user_df = pd.DataFrame([user_vals], columns=feature_columns)

            # Scale user input (Pass DataFrame directly to scaler to resolve UserWarning)
            user_scaled = scaler.transform(user_df)

            # Predict cluster using the KNeighborsClassifier proxy model
            # Note: If model is the raw DBSCAN object (due to 0 clusters), it should predict -1.
            if isinstance(model, DBSCAN):
                 cluster = -1 # Predict -1 if no proxy was trained
            else:
                 cluster = model.predict(user_scaled)[0]

            cluster_result = f"Cluster: {int(cluster)}"
            if cluster == -1:
                cluster_result = "Noise/Outlier (-1)"

            print("\n" + "="*30)
            print(f"**[RESULT] Assigned {cluster_result}**")
            print("="*30)

    return predict_loop


if __name__ == "__main__":
    # Updated CLI checks to match the deployment standard structure
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage (Train/Predict): python script_name.py <dataset_path> <pkl_filename>")
        print("Usage (Predict Only): python script_name.py <pkl_filename> predict_only")
        sys.exit(1)
        
    if len(sys.argv) == 3:
        if sys.argv[2].lower() == 'predict_only':
            # Transform Only Mode
            pkl_filename = sys.argv[1]
            try:
                # Use dummy path/None value
                predict_func = train_and_save_dbscan(None, pkl_filename) 
                predict_func()
            except Exception as e:
                print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Standard Training and Prediction
            dataset_path = sys.argv[1]
            pkl_filename = sys.argv[2]
            try:
                predict_func = train_and_save_dbscan(dataset_path, pkl_filename)
                predict_func()
            except Exception as e:
                print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
                sys.exit(1)
    
    elif len(sys.argv) == 2:
        # Prediction Only Mode shortcut (python script_name.py pkl_filename)
        pkl_filename = sys.argv[1]
        try:
            predict_func = train_and_save_dbscan(None, pkl_filename)
            predict_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)