import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load

from sklearn.mixture import BayesianGaussianMixture # Correct Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import warnings

# Define a reasonable range for components (clusters)
DEFAULT_N_COMPONENTS = 8

def train_and_save_bgmm(dataset_path: str, pkl_filename: str):
    """
    Trains a Bayesian Gaussian Mixture Model (BGMM), saves necessary components, 
    and returns a predict function.
    
    The target_column argument is removed as clustering is unsupervised.
    """
    
    # Suppress warnings related to convergence and initialization
    warnings.filterwarnings("ignore", category=UserWarning)

    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    model_path = os.path.join(trained_dir, pkl_filename)

    # Variables to store components for saving/loading
    model = None
    scaler = None
    feature_columns = None
    encoders = {}

    ## --- CASE 1: Load existing model ---
    if os.path.exists(model_path):
        print(f"**[INFO] Loading existing Bayesian Gaussian Mixture model '{pkl_filename}'...**")
        # Load the model, scaler, feature columns, and encoders
        model, scaler, feature_columns, encoders = load(model_path)
        print("[INFO] Model loaded successfully.")

    ## --- CASE 2: Train and Save a new model ---
    else:
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
                # Fallback attempt for CSV-like files
                data = pd.read_csv(dataset_path, sep=None, engine='python')
        except Exception as e:
            # Added more robust error handling for data loading
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

        # 3. SCALING
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Keep track of feature names
        feature_columns = list(X.columns)

        ## --- TRAIN BAYESIAN GAUSSIAN MIXTURE MODEL ---
        print(f"\n**[INFO] Training BayesianGaussianMixture with n_components={DEFAULT_N_COMPONENTS}...**")
        
        # Initialize and fit the BGMM
        # BGMM allows the model to effectively ignore components if the data 
        # doesn't support them, often resulting in fewer effective clusters than n_components.
        model = BayesianGaussianMixture(
            n_components=DEFAULT_N_COMPONENTS, 
            covariance_type='full', # Standard choice
            random_state=42,
            max_iter=500
        )

        # Fit the model and get the cluster assignments (labels)
        labels = model.fit_predict(X_scaled)

        ## --- EVALUATION ---
        # Only evaluate if there is more than one unique label found
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            print(f"\n[INFO] Silhouette Score (effective components): **{score:.4f}**")
        else:
             print("[INFO] Only 1 effective cluster found. Silhouette Score not calculated.")

        ## --- SAVE MODEL COMPONENTS ---
        # Save the BGMM model directly, as it has a .predict() method
        dump((model, scaler, feature_columns, encoders), model_path)
        print(f"\n[INFO] Bayesian Gaussian Mixture Model saved as '{pkl_filename}'")
    
    
    ## --- PREDICT FUNCTION (for new data inference) ---
    def predict():
        """
        Prompts the user for feature values, preprocesses the input, 
        and prints the component assignment.
        """
        if not model:
            print("[ERROR] Model not loaded/trained. Cannot predict.", file=sys.stderr)
            return

        print("\n" + "="*50)
        print("ENTER FEATURE VALUES FOR CLUSTER PREDICTION (BGMM)")
        print("="*50)
        user_input_dict = {}

        for col in feature_columns:
            while True:
                try:
                    value = input(f"Enter value for **{col}**: ")
                    
                    # Handle categorical columns
                    if col in encoders:
                        le = encoders[col]
                        try:
                            encoded_value = le.transform([str(value).strip()])[0]
                            user_input_dict[col] = encoded_value
                            break
                        except ValueError:
                            print(f"[ERROR] Value '{value}' for column '{col}' is not a known category. Please re-enter.")
                            continue
                    
                    # Handle numerical columns
                    else:
                        user_input_dict[col] = float(value)
                        break
                except ValueError:
                    print(f"[ERROR] Invalid number format for column '{col}'. Please enter a numeric value.")
                    continue
        
        # Convert user input to a DataFrame, ensuring column order
        user_df = pd.DataFrame([user_input_dict], columns=feature_columns)

        # Scale user input
        user_scaled = scaler.transform(user_df)

        # Predict component assignment using the BGMM's native .predict() method
        cluster = model.predict(user_scaled)[0]

        print("\n" + "="*50)
        print(f"**[RESULT] Assigned Component/Cluster: {cluster}**")
        print("="*50)

    return predict


if __name__ == "__main__":
    # Check for correct number of arguments (dataset_path, pkl_filename)
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <dataset_path> <pkl_filename>")
        print("Note: The target_column argument is removed for clustering problems.")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    pkl_filename = sys.argv[2]
    
    try:
        # train_and_save_bgmm returns the prediction function
        predict_func = train_and_save_bgmm(dataset_path, pkl_filename)
        # Call the prediction function
        predict_func()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)