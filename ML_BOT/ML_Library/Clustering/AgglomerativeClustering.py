import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from joblib import dump, load
from io import StringIO
import os
import sys


from sklearn.neighbors import KNeighborsClassifier 
# ... (rest of the code)

# Define a reasonable default number of clusters if not optimized
DEFAULT_N_CLUSTERS = 8

def train_and_save_agglomerative(dataset_path: str, pkl_filename: str):
    """
    Trains an Agglomerative Clustering model, saves necessary components, 
    and returns a predict function.
    
    NOTE: The target_column argument is removed as clustering is unsupervised.
    """

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
        print(f"**[INFO] Loading existing Agglomerative model '{pkl_filename}'...**")
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

        # X is the entire dataset (no target column in unsupervised learning)
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
        # Using mean for numerical columns.
        X.fillna(X.mean(numeric_only=True), inplace=True)

        # 3. SCALING
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Keep track of feature names
        feature_columns = list(X.columns)

        ## --- TRAIN MODEL (Agglomerative Clustering) ---
        print(f"\n**[INFO] Training AgglomerativeClustering with n_clusters={DEFAULT_N_CLUSTERS}...**")
        
        # NOTE on AgglomerativeClustering for production inference:
        # AgglomerativeClustering does not have a 'predict' method.
        # To enable inference on new data, we must use a model that supports
        # the 'predict' or 'kneighbors'/'radius_neighbors' method for new points.
        # A common proxy is using Nearest Neighbors post-clustering.
        
        # We will fit the model just to get the cluster labels on the training data.
        model = AgglomerativeClustering(
            n_clusters=DEFAULT_N_CLUSTERS,
            linkage="ward",      # ward requires euclidean metric
            metric="euclidean"   # metric parameter is now common
        )

        # Fit and assign labels to the training data
        train_labels = model.fit_predict(X_scaled)
        
        # For inference on new data, we need a proxy prediction mechanism.
        # Since Agglomerative doesn't have .predict(), we'll use a NearestNeighbors
        # classifier trained on the resulting labels of the training data.
        from sklearn.neighbors import KNeighborsClassifier
        
        # Train a K-Nearest Neighbors classifier using the scaled training data 
        # and the cluster labels found by AgglomerativeClustering.
        # This acts as our 'prediction' model for new data.
        predictive_model = KNeighborsClassifier(n_neighbors=5)
        predictive_model.fit(X_scaled, train_labels)
        
        # Overwrite the original model variable to save the predictive model
        model = predictive_model

        ## --- EVALUATION ---
        # The silhouette score is the standard for clustering evaluation
        score = silhouette_score(X_scaled, train_labels)
        print(f"\n[INFO] Silhouette Score (n_clusters={DEFAULT_N_CLUSTERS}): **{score:.4f}**")

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
            print("[ERROR] Model not loaded/trained. Cannot predict.", file=sys.stderr)
            return

        print("\n" + "="*50)
        print("ENTER FEATURE VALUES FOR CLUSTER PREDICTION")
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
                            # Transform the new category (using .astype(str) for consistency)
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
        
        # Convert user input to a DataFrame, ensuring column order matches training data
        # Note: If a numerical column received NA/NaN during training, it won't be filled here, 
        # but since we take input, we assume all fields are present.
        user_df = pd.DataFrame([user_input_dict], columns=feature_columns)

        # Scale user input using the **SAVED** scaler
        user_scaled = scaler.transform(user_df)

        # Predict cluster using the KNeighborsClassifier proxy model
        cluster = model.predict(user_scaled)[0]

        print("\n" + "="*50)
        print(f"**[RESULT] Assigned Cluster: {cluster}**")
        print("="*50)

    return predict


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <dataset_path> <pkl_filename>")
        print("Note: The target_column argument is removed for clustering problems.")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    pkl_filename = sys.argv[2]
    
    try:
        # train_and_save_agglomerative returns the prediction function
        predict_func = train_and_save_agglomerative(dataset_path, pkl_filename)
        # Call the prediction function
        predict_func()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)