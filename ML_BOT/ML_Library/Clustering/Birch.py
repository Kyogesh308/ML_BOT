import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load

from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import warnings

# Define a reasonable number of clusters (Birch's n_clusters is often set high 
# to allow it to find the natural groupings, or set to None for only tree building)
DEFAULT_N_CLUSTERS = 5 
# Recommended parameters for Birch
DEFAULT_THRESHOLD = 0.5 

def train_and_save_birch(dataset_path: str, pkl_filename: str):
    """
    Trains a Birch Clustering model, saves necessary components, 
    and returns a predict function.
    
    The target_column argument is removed as clustering is unsupervised.
    """
    
    # Ignore common runtime warnings (e.g., related to feature calculation)
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
        print(f"**[INFO] Loading existing Birch model '{pkl_filename}'...**")
        # Ensure we load all four saved components
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
                # Use .astype(str) to handle potential mixed types or NaNs gracefully
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

        # 2. Fill NA (Imputation)
        # Only use mean for columns that are numerically type-castable
        X.fillna(X.mean(numeric_only=True), inplace=True)

        # 3. SCALING
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Keep track of feature names
        feature_columns = list(X.columns)

        ## --- TRAIN BIRCH MODEL ---
        print(f"\n**[INFO] Training Birch Clustering with n_clusters={DEFAULT_N_CLUSTERS} and threshold={DEFAULT_THRESHOLD}...**")
        
        # Initialize and fit the Birch model
        model = Birch(
            n_clusters=DEFAULT_N_CLUSTERS, 
            threshold=DEFAULT_THRESHOLD,
            branching_factor=50 # Default, good for most cases
        )

        # Fit the model and get the cluster assignments (labels)
        labels = model.fit_predict(X_scaled)

        ## --- EVALUATION ---
        if len(np.unique(labels)) > 1:
            # Silhouette Score is the standard metric for clustering evaluation
            score = silhouette_score(X_scaled, labels)
            print(f"\n[INFO] Silhouette Score: **{score:.4f}**")
        else:
             print("[INFO] Only 1 effective cluster found. Silhouette Score not calculated.")

        ## --- SAVE MODEL COMPONENTS ---
        # Save the model, scaler, feature names, and encoders
        dump((model, scaler, feature_columns, encoders), model_path)
        print(f"\n[INFO] Birch Model saved as '{pkl_filename}'")
    
    
    ## --- PREDICT FUNCTION (for new data inference) ---
    def predict():
        """
        Prompts the user for feature values, preprocesses the input, 
        and prints the cluster assignment.
        """
        if not model:
            print("[ERROR] Model not loaded/trained. Cannot predict.", file=sys.stderr)
            return

        print("\n" + "="*50)
        print("ENTER FEATURE VALUES FOR CLUSTER PREDICTION (BIRCH)")
        print("="*50)
        user_input_dict = {}

        for col in feature_columns:
            while True:
                try:
                    value = input(f"Enter value for **{col}**: ")
                    
                    # Handle categorical columns using the saved encoders
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
        
        # Convert user input to a DataFrame, ensuring column order matches training
        # We use .to_numpy() to get the required shape for scaling
        user_df = pd.DataFrame([user_input_dict], columns=feature_columns)

        # Scale user input
        user_scaled = scaler.transform(user_df.to_numpy())

        # Predict cluster using the Birch's native .predict() method
        cluster = model.predict(user_scaled)[0]

        print("\n" + "="*50)
        print(f"**[RESULT] Assigned Cluster: {cluster}**")
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
        # train_and_save_birch returns the prediction function
        predict_func = train_and_save_birch(dataset_path, pkl_filename)
        # Call the prediction function
        predict_func()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)