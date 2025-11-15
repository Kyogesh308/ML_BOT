import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load

# --- Correct Clustering Imports ---
from sklearn.mixture import GaussianMixture # The proper GMM clustering model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import warnings

# --- Constants ---
DEFAULT_N_COMPONENTS = 5 # Standard number of components (clusters)

def train_and_save_model(dataset_path: str, pkl_filename: str):
    """
    Trains a Gaussian Mixture Model (GMM) for clustering, saves necessary components, 
    and returns a predict function.
    
    The target_column argument is removed as GMM is unsupervised clustering.
    """
    
    warnings.filterwarnings("ignore", category=UserWarning)

    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    model_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    model = None # GMM Model
    scaler = None
    feature_columns = None
    encoders = {}

    ## --- CASE 1: Load existing model ---
    if os.path.exists(model_path):
        print(f"**[INFO] Loading existing Gaussian Mixture Model '{pkl_filename}'...**")
        # Load the four required components
        model, scaler, feature_columns, encoders = load(model_path)
        print("[INFO] Model loaded successfully.")

    ## --- CASE 2: Train and Save a new model ---
    else:
        # --- LOAD DATA (Restored robust multi-file loading logic) ---
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

        # X is the entire dataset (unsupervised)
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

        # --- TRAIN GAUSSIAN MIXTURE MODEL ---
        print(f"\n**[INFO] Training Gaussian Mixture Model with n_components={DEFAULT_N_COMPONENTS}...**")
        
        model = GaussianMixture(
            n_components=DEFAULT_N_COMPONENTS,
            covariance_type='full', # Standard choice
            random_state=42
        )
        labels = model.fit_predict(X_scaled) # Fit and predict labels on training data

        # --- EVALUATION ---
        if len(np.unique(labels)) > 1:
            # Silhouette Score is the standard metric for clustering
            score = silhouette_score(X_scaled, labels)
            print(f"Training completion: Silhouette Score = {score:.4f}")
        else:
            print("Training completed with only 1 effective cluster. Silhouette Score not calculated.")

        # --- SAVE THE MODEL COMPONENTS ---
        # Save the GMM model, scaler, feature names, and encoders
        dump((model, scaler, feature_columns, encoders), model_path)
        print(f"Model saved as '{pkl_filename}'")

    # --- Prediction function ---
    def predict():
        if not model:
            print("[ERROR] Model not loaded/trained. Cannot predict.", file=sys.stderr)
            return
            
        print("\n" + "="*50)
        print("ENTER FEATURE VALUES FOR CLUSTER PREDICTION (GMM)")
        print("="*50)
        user_input_dict = {}

        for col in feature_columns:
            while True:
                try:
                    value = input(f"Enter value for **{col}**: ")
                    
                    if col in encoders:
                        le = encoders[col]
                        try:
                            encoded_value = le.transform([str(value).strip()])[0]
                            user_input_dict[col] = encoded_value
                            break
                        except ValueError:
                            print(f"[ERROR] Value '{value}' for column '{col}' is not a known category. Please re-enter.")
                            continue
                    else:
                        user_input_dict[col] = float(value)
                        break
                except ValueError:
                    print(f"[ERROR] Invalid number format for column '{col}'. Please enter a numeric value.")
                    continue

        user_df = pd.DataFrame([user_input_dict], columns=feature_columns)
        
        # FIX: Convert DataFrame to NumPy array and reshape only if necessary (GMM expects 2D array)
        user_scaled = scaler.transform(user_df.to_numpy())
        
        # Use GMM's native .predict() method
        pred = model.predict(user_scaled)[0]
        
        print("\n" + "="*50)
        print(f"**[RESULT] Assigned Cluster/Component: {pred}**")
        print("="*50)

    return predict

if __name__ == "__main__":
    # Check for correct number of arguments (script_name, dataset_path, pkl_filename)
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <dataset_path> <pkl_filename>")
        print("Note: The target_column argument is removed for clustering problems.")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    pkl_filename = sys.argv[2]
    
    try:
        predict_func = train_and_save_model(dataset_path, pkl_filename)
        predict_func()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)