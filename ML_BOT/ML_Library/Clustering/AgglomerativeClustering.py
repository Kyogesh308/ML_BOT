import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier 
import warnings

# --- Constants ---
DEFAULT_K_RANGE = range(3, 11) # Range of K to test (3 to 10)
DEFAULT_PROXY_K = 5 
DEFAULT_LINKAGE = "ward"

def find_optimal_k_agg(X_scaled, k_range):
    """Calculates Silhouette Score for Agglomerative Clustering across a range of K and returns the optimal K."""
    best_score = -1
    optimal_k = k_range.start
    
    print(f"[INFO] Searching for optimal K using Silhouette Score across {list(k_range)}...")
    
    for k in k_range:
        try:
            # AgglomerativeClustering requires metric and linkage
            model = AgglomerativeClustering(n_clusters=k, linkage=DEFAULT_LINKAGE, metric="euclidean")
            
            labels = model.fit_predict(X_scaled)
            
            # Skip if only 1 cluster or too few points are found
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    optimal_k = k
            
        except Exception as e:
            # Catch exceptions like n_clusters > n_samples/2 or matrix errors
            # print(f"[WARNING] Skipping K={k} due to error: {e}")
            continue
            
    print(f"[INFO] Optimal K found: {optimal_k} (Silhouette Score: {best_score:.4f})")
    return optimal_k


def train_and_save_agglomerative(dataset_path: str, pkl_filename: str):
    """
    Trains an Agglomerative Clustering model using the optimal K determined by Silhouette Score, 
    fits a KNeighborsClassifier proxy, and saves the necessary components for prediction.
    """

    # Ignore common convergence warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables to store components for saving/loading (5 components for clustering with optimal_k)
    model = None          # Stores the KNeighborsClassifier proxy
    scaler = None
    feature_columns = None
    encoders = {}
    optimal_k = None
    is_trained = False

    ## --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing Agglomerative model proxy '{pkl_filename}'...**")
        try:
            # Load the five required components (Model, Scaler, Features, Encoders, Optimal_K)
            model, scaler, feature_columns, encoders, optimal_k = load(full_path)
            print(f"[INFO] Model loaded successfully. Optimal K: {optimal_k}")
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

        ## --- OPTIMAL K SELECTION (NEW STEP) ---
        optimal_k = find_optimal_k_agg(X_scaled, DEFAULT_K_RANGE)

        ## --- TRAIN FINAL AGGLOMERATIVE CLUSTERING ---
        print(f"\n**[INFO] Training FINAL AgglomerativeClustering with K={optimal_k}...**")
        
        # 1. Fit AgglomerativeClustering to get the final labels
        ac_model = AgglomerativeClustering(
            n_clusters=optimal_k,
            linkage=DEFAULT_LINKAGE, 
            metric="euclidean" 
        )

        train_labels = ac_model.fit_predict(X_scaled)
        
        # 2. Train KNC as the proxy prediction model
        print(f"[INFO] Fitting KNeighborsClassifier proxy (K={DEFAULT_PROXY_K}) for inference...")
        predictive_model = KNeighborsClassifier(n_neighbors=DEFAULT_PROXY_K, n_jobs=-1)
        predictive_model.fit(X_scaled, train_labels)
        
        # Overwrite the original model variable to save the predictive model
        model = predictive_model

        ## --- EVALUATION ---
        if optimal_k > 1:
            score = silhouette_score(X_scaled, train_labels)
            print(f"\n[INFO] Final Silhouette Score (K={optimal_k}): **{score:.4f}**")
        else:
            print("[INFO] Final model resulted in 1 cluster. Silhouette Score not calculated.")

        ## --- SAVE MODEL COMPONENTS ---
        # Save the predictive model (KNeighbors), scaler, feature names, encoders, and optimal_k
        dump((model, scaler, feature_columns, encoders, optimal_k), full_path)
        print(f"\n[INFO] Predictive model saved as '{pkl_filename}' (K={optimal_k})")
        is_trained = True
    
    
    ## --- PREDICT FUNCTION (Unlimited Loop for new data inference) ---
    def predict_loop():
        nonlocal model, scaler, feature_columns, encoders, optimal_k
        
        if model is None:
            try:
                # Load the five required components
                model, scaler, feature_columns, encoders, optimal_k = load(full_path)
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model for prediction: {e}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"**Starting UNLIMITED Cluster Prediction Mode for '{pkl_filename}'**")
        print(f"Model uses {optimal_k} optimized clusters.")
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
            user_df = pd.DataFrame([user_vals], columns=feature_columns)
            
            # Apply scaling (Pass DataFrame directly to scaler to resolve UserWarning)
            user_scaled = scaler.transform(user_df)

            # Predict cluster using the KNeighborsClassifier proxy model
            cluster = model.predict(user_scaled)[0]

            print("\n" + "="*30)
            print(f"**[RESULT] Assigned Cluster: {int(cluster)}**")
            print("="*30)

    return predict_loop


if __name__ == "__main__":
    # Updated CLI checks to match the deployment standard structure
    # Expects: dataset_path, pkl_filename (2 args for train/predict)
    
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
                predict_func = train_and_save_agglomerative(None, pkl_filename) 
                predict_func()
            except Exception as e:
                print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Standard Training and Prediction
            dataset_path = sys.argv[1]
            pkl_filename = sys.argv[2]
            try:
                predict_func = train_and_save_agglomerative(dataset_path, pkl_filename)
                predict_func()
            except Exception as e:
                print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
                sys.exit(1)
    
    elif len(sys.argv) == 2:
        # Prediction Only Mode shortcut (python script_name.py pkl_filename)
        pkl_filename = sys.argv[1]
        try:
            predict_func = train_and_save_agglomerative(None, pkl_filename)
            predict_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)