import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import warnings

# --- Constants ---
DEFAULT_K_RANGE = range(3, 11) # Range of K to test (3 to 10)
DEFAULT_MAX_ITER = 500
DEFAULT_COV_TYPE = 'full'

def find_optimal_k_bgmm(X_scaled, k_range):
    """Calculates Silhouette Score for Bayesian Gaussian Mixture across a range of K and returns the optimal K."""
    best_score = -1
    optimal_k = k_range.start
    
    print(f"[INFO] Searching for optimal K using Silhouette Score across {list(k_range)}...")
    
    for k in k_range:
        try:
            # BGMM initialization for testing K
            model = BayesianGaussianMixture(
                n_components=k, 
                covariance_type=DEFAULT_COV_TYPE,
                random_state=42,
                max_iter=DEFAULT_MAX_ITER
            )
            
            labels = model.fit_predict(X_scaled)
            
            # Use only samples from effective components (weight > 0.01) for robust scoring
            effective_components = (model.weights_ > 0.01).sum()
            
            if effective_components > 1:
                # Use only samples assigned to effective components if possible
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    optimal_k = k
            
        except Exception as e:
            # Catch exceptions during fitting (e.g., singular matrix due to bad k)
            # print(f"[WARNING] Skipping K={k} due to error: {e}")
            continue
            
    print(f"[INFO] Optimal K found: {optimal_k} (Silhouette Score: {best_score:.4f})")
    return optimal_k


def train_and_save_bgmm(dataset_path: str, pkl_filename: str):
    """
    Trains a Bayesian Gaussian Mixture Model (BGMM) using the optimal K determined by Silhouette Score, 
    and saves the necessary components for prediction.
    """
    
    # Suppress warnings related to convergence and initialization
    warnings.filterwarnings("ignore", category=UserWarning)

    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables to store components for saving/loading
    model = None
    scaler = None
    feature_columns = None
    encoders = {}
    optimal_k = None # New component to store the result of the K search
    is_trained = False

    ## --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing Bayesian Gaussian Mixture model '{pkl_filename}'...**")
        try:
            # Load the five required components
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

        # 3. SCALING (CRITICAL for distance/probabilistic models like BGMM)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        feature_columns = list(X.columns)

        ## --- OPTIMAL K SELECTION (NEW STEP) ---
        optimal_k = find_optimal_k_bgmm(X_scaled, DEFAULT_K_RANGE)

        ## --- TRAIN FINAL BGMM MODEL ---
        print(f"\n**[INFO] Training FINAL BayesianGaussianMixture with K={optimal_k}...**")
        
        # Fit model using the optimal K determined by the search
        model = BayesianGaussianMixture(
            n_components=optimal_k, 
            covariance_type=DEFAULT_COV_TYPE,
            random_state=42,
            max_iter=DEFAULT_MAX_ITER
        )

        labels = model.fit_predict(X_scaled)
        
        ## --- EVALUATION ---
        effective_components = (model.weights_ > 0.01).sum()
        print(f"[INFO] Effective components found: {effective_components}")
        
        if effective_components > 1:
            score = silhouette_score(X_scaled, labels)
            print(f"\n[INFO] Final Silhouette Score (K={optimal_k}): **{score:.4f}**")
        else:
            print("[INFO] Final model resulted in 1 effective component. Silhouette Score not calculated.")

        ## --- SAVE MODEL COMPONENTS ---
        # Save the BGMM model, scaler, feature names, encoders, and optimal_k
        dump((model, scaler, feature_columns, encoders, optimal_k), full_path)
        print(f"\n[INFO] Bayesian Gaussian Mixture Model saved as '{pkl_filename}' (K={optimal_k})")
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
            
            # Apply scaling (resolves UserWarning)
            user_scaled = scaler.transform(user_df)

            # Predict component assignment using the BGMM's native .predict() method
            cluster = model.predict(user_scaled)[0]

            print("\n" + "="*30)
            print(f"**[RESULT] Assigned Component/Cluster: {int(cluster)}**")
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
                predict_func = train_and_save_bgmm(None, pkl_filename) 
                predict_func()
            except Exception as e:
                print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Standard Training and Prediction
            dataset_path = sys.argv[1]
            pkl_filename = sys.argv[2]
            try:
                predict_func = train_and_save_bgmm(dataset_path, pkl_filename)
                predict_func()
            except Exception as e:
                print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
                sys.exit(1)
    
    elif len(sys.argv) == 2:
        # Prediction Only Mode shortcut (python script_name.py pkl_filename)
        pkl_filename = sys.argv[1]
        try:
            predict_func = train_and_save_bgmm(None, pkl_filename)
            predict_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)