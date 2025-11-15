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

# Define standard DBSCAN parameters
DEFAULT_EPS = 0.5    # The maximum distance between two samples for one to be considered as in the neighborhood of the other
DEFAULT_MIN_SAMPLES = 5 # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point

def train_and_save_dbscan(dataset_path: str, pkl_filename: str):
    """
    Trains a DBSCAN model, uses a KNeighborsClassifier as a proxy for prediction, 
    saves necessary components, and returns a predict function.
    
    The target_column argument is removed as clustering is unsupervised.
    """
    
    warnings.filterwarnings("ignore", category=UserWarning)

    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    model_path = os.path.join(trained_dir, pkl_filename)

    # Variables to store components for saving/loading
    model = None          # This will be the KNeighborsClassifier proxy
    scaler = None
    feature_columns = None
    encoders = {}

    ## --- CASE 1: Load existing model ---
    if os.path.exists(model_path):
        print(f"**[INFO] Loading existing DBSCAN predictive model '{pkl_filename}'...**")
        # Ensure we load all four saved components
        model, scaler, feature_columns, encoders = load(model_path)
        print("[INFO] Predictive model loaded successfully.")

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

        ## --- TRAIN DBSCAN MODEL ---
        print(f"\n**[INFO] Training DBSCAN Clustering with eps={DEFAULT_EPS} and min_samples={DEFAULT_MIN_SAMPLES}...**")
        
        # Initialize and fit the DBSCAN model
        dbscan_model = DBSCAN(
            eps=DEFAULT_EPS, 
            min_samples=DEFAULT_MIN_SAMPLES
        )

        # Fit the model and retrieve the cluster labels
        dbscan_model.fit(X_scaled)
        labels = dbscan_model.labels_

        # DBSCAN assigns -1 to noise/outliers. We only consider the actual clusters (label >= 0).
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[INFO] Estimated number of clusters: {n_clusters}")

        ## --- CREATE PREDICTION PROXY (MANDATORY for DBSCAN inference) ---
        # DBSCAN has no .predict() method. We use KNeighborsClassifier trained 
        # on the scaled data and the resulting DBSCAN labels.
        # Note: KNN can assign a new point to a noise cluster (-1), which is correct behavior.
        if n_clusters > 0:
            print("[INFO] Creating KNeighborsClassifier proxy for prediction...")
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_scaled, labels)
        else:
            print("[WARNING] DBSCAN found 0 clusters (all data is noise). Cannot create a meaningful prediction proxy.")
            # Still save components, but model will predict -1 or the single noise cluster

        ## --- EVALUATION ---
        # We must filter out noise points (-1) for a valid silhouette score.
        mask = (labels != -1)
        if np.sum(mask) > 1 and n_clusters > 1:
            score = silhouette_score(X_scaled[mask], labels[mask])
            print(f"\n[INFO] Silhouette Score (excluding noise): **{score:.4f}**")
        else:
             print("[INFO] Silhouette Score not calculated (too few points or clusters).")

        ## --- SAVE MODEL COMPONENTS ---
        # Save the predictive model (KNeighbors), scaler, feature names, and encoders
        dump((model, scaler, feature_columns, encoders), model_path)
        print(f"\n[INFO] Predictive model (KNeighborsClassifier) saved as '{pkl_filename}'")
    
    
    ## --- PREDICT FUNCTION (for new data inference) ---
    def predict():
        """
        Prompts the user for feature values, preprocesses the input, 
        and prints the cluster assignment.
        """
        if not model:
            print("[ERROR] Prediction proxy model not loaded/trained. Cannot predict.", file=sys.stderr)
            return

        print("\n" + "="*50)
        print("ENTER FEATURE VALUES FOR CLUSTER PREDICTION (DBSCAN)")
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
        
        # Convert user input to a DataFrame
        user_df = pd.DataFrame([user_input_dict], columns=feature_columns)

        # Scale user input
        user_scaled = scaler.transform(user_df.to_numpy())

        # Predict cluster using the KNeighborsClassifier proxy model
        cluster = model.predict(user_scaled)[0]
        
        cluster_result = f"Cluster: {cluster}"
        if cluster == -1:
            cluster_result = "Noise/Outlier (-1)"

        print("\n" + "="*50)
        print(f"**[RESULT] Assigned {cluster_result}**")
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
        # train_and_save_dbscan returns the prediction function
        predict_func = train_and_save_dbscan(dataset_path, pkl_filename)
        # Call the prediction function
        predict_func()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)