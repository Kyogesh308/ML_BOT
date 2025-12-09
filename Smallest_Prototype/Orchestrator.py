import os
import pandas as pd
import string
import sys
import numpy as np

# FIX: Import the model_suggestion function to return problem_type_llm
from ModelSuggestion import model_suggestion
from ModelGenerator import generate_script_content, extract_code_block, save_script, execute_script
from io import StringIO

# --- MASTER TEMPLATE STORAGE ---
# These templates reflect the fixed, robust structure we built together.

REGRESSION_TEMPLATE = r"""
import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# --- Constants ---
DEFAULT_TEST_SIZE = 0.2

def train_and_save_model(dataset_path: str, target_column: str, pkl_filename: str):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    model = None
    scaler = None
    feature_columns = None
    encoders = {}
    is_trained = False

    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing {{model_name}} model '{{pkl_filename}}'...**")
        try:
            model, scaler, feature_columns, encoders = load(full_path)
            print("[INFO] Model loaded successfully.")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Could not load model from '{{full_path}}' due to error: {{e}}. Retraining...")
            pass
    
    if not is_trained:
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
            print(f"[ERROR] Failed to load data: {{e}}", file=sys.stderr)
            sys.exit(1)
            
        if target_column not in data.columns:
            raise KeyError(f"Target column '{{target_column}}' not found in the dataset.")

        y = data[target_column]
        X = data.drop(target_column, axis=1)

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
        
        # --- SPLIT DATA ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=DEFAULT_TEST_SIZE, random_state=42
        )

        # --- TRAIN MODEL ---
        print(f"\n**[INFO] Training {{model_name}}...**")
        
        # NOTE: Model initialization is dynamically injected here.
        model = {{model_name}}()
        model.fit(X_train, y_train)

        # --- EVALUATION ---
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"\n[INFO] R-squared Score (Test): **{{r2:.4f}}**")
        print(f"[INFO] Mean Squared Error (Test): **{{mse:.4f}}**")

        # --- SAVE MODEL ---
        dump((model, scaler, feature_columns, encoders), full_path)
        print(f"[INFO] Model saved as '{{pkl_filename}}'")
        is_trained = True

    # --- Prediction function (Unlimited Loop) ---
    def predict_loop():
        nonlocal model, scaler, feature_columns, encoders
        
        if model is None:
            try:
                model, scaler, feature_columns, encoders = load(full_path)
            except Exception as e:
                print(f"[ERROR] Failed to load model for prediction: {{e}}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"**Starting UNLIMITED Prediction Mode for '{{pkl_filename}}'**")
        print("Type 'exit' or 'quit' to close the program.")
        print("="*70)
        
        while True:
            user_vals = []
            print("\n" + "-"*30)
            
            should_exit = False
            for col in feature_columns:
                while True:
                    try:
                        val = input(f"Enter value for **{{col}}**: ")
                        input_val_str = str(val).strip()

                        if input_val_str.lower() in ['exit', 'quit']:
                            should_exit = True
                            break
                        
                        if col in encoders:
                            le = encoders[col]
                            try:
                                encoded_val = le.transform([input_val_str])[0]
                                user_vals.append(encoded_val)
                                break
                            except ValueError:
                                known_classes = list(le.classes_)
                                if len(known_classes) > 10:
                                    known_classes = known_classes[:10] + ["..."]
                                print(f"[ERROR] Value '{{input_val_str}}' for column '{{col}}' is not a known category. Known examples: {{known_classes}}")
                                continue
                        else:
                            user_vals.append(float(val))
                            break
                    except ValueError:
                        print(f"[ERROR] Invalid number format for column '{{col}}'. Please enter a numeric value.")
                        continue
                
                if should_exit:
                    break
            
            if should_exit:
                print("\n**[INFO] Prediction mode terminated by user.**")
                break

            df_input = pd.DataFrame([user_vals], columns=feature_columns)
            user_scaled = scaler.transform(df_input)

            pred = model.predict(user_scaled)[0]
            
            print("\n" + "="*30)
            print(f"**[RESULT] Predicted Continuous Outcome: {{pred:.4f}}**")
            print("="*30)

    return predict_loop

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage (Train/Predict): python script_name.py <dataset_path> <target_column> <pkl_filename>")
        print("Usage (Predict Only): python script_name.py <pkl_filename> predict_only")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    target_column = sys.argv[2]
    pkl_filename = sys.argv[3]
    
    try:
        predict_func = train_and_save_model(dataset_path, target_column, pkl_filename)
        predict_func()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {{e}}", file=sys.stderr)
        sys.exit(1)
"""

CLASSIFICATION_TEMPLATE = r"""
import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score 
from sklearn.ensemble import RandomForestClassifier # FIX: Add common base model import

# --- Constants ---
DEFAULT_TEST_SIZE = 0.2
# FIX: Dynamic import of the suggested model (will be substituted in Orchestrator)
from sklearn.linear_model import LogisticRegression 

def train_and_save_model(dataset_path: str, target_column: str, pkl_filename: str):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    model = None
    scaler = None
    feature_columns = None
    feature_encoders = {}
    target_encoder = None
    is_trained = False

    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing {{model_name}} model '{{pkl_filename}}'...**")
        try:
            model, scaler, feature_columns, feature_encoders, target_encoder = load(full_path) 
            print("[INFO] Model loaded successfully.")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Could not load model from '{{full_path}}' due to error: {{e}}. Retraining...")
            pass
            
    if not is_trained:
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
            print(f"[ERROR] Failed to load data: {{e}}", file=sys.stderr)
            sys.exit(1)
            
        if target_column not in data.columns:
            raise KeyError(f"Target column '{{target_column}}' not found in the dataset.")

        y_raw = data[target_column].astype(str)
        X = data.drop(target_column, axis=1)

        # --- PREPROCESSING ---

        # 1. Encode Target Variable (y)
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw)

        # 2. Encode Feature Columns (X)
        for col in X.columns:
            if X[col].dtype == object:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                feature_encoders[col] = le
        
        # 3. Fill NA (Imputation)
        X.fillna(X.mean(numeric_only=True), inplace=True)

        # 4. SCALING
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_columns = list(X.columns)
        
        # --- SPLIT DATA ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=DEFAULT_TEST_SIZE, random_state=42
        )

        # --- TRAIN CLASSIFIER ---
        print(f"\n**[INFO] Training {{model_name}} Classifier...**")
        
        # NOTE: Model initialization is dynamically injected here.
        model = {{model_name}}()
        model.fit(X_train, y_train)

        # --- EVALUATION ---
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n[INFO] Accuracy Score (Test): **{{acc:.4f}}**")
        print(f"[INFO] F1 Score (Weighted Test): **{{f1:.4f}}**")

        # --- SAVE MODEL ---
        dump((model, scaler, feature_columns, feature_encoders, target_encoder), full_path)
        print(f"[INFO] Model saved as '{{pkl_filename}}'")
        is_trained = True

    # --- Prediction function (Unlimited Loop) ---
    def predict_loop():
        nonlocal model, scaler, feature_columns, feature_encoders, target_encoder
        
        if model is None:
            try:
                model, scaler, feature_columns, feature_encoders, target_encoder = load(full_path)
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model for prediction: {{e}}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"**Starting UNLIMITED Prediction Mode for '{{pkl_filename}}'**")
        print("Type 'exit' or 'quit' to close the program.")
        print("="*70)
        
        while True:
            user_vals = []
            print("\n" + "-"*30)
            
            should_exit = False
            for col in feature_columns:
                while True:
                    try:
                        val = input(f"Enter value for **{{col}}**: ")
                        input_val_str = str(val).strip()

                        if input_val_str.lower() in ['exit', 'quit']:
                            should_exit = True
                            break
                            
                        if col in feature_encoders:
                            le = feature_encoders[col]
                            try:
                                encoded_val = le.transform([input_val_str])[0]
                                user_vals.append(encoded_val)
                                break
                            except ValueError:
                                known_classes = list(le.classes_)
                                if len(known_classes) > 10:
                                    known_classes = known_classes[:10] + ["..."]
                                print(f"[ERROR] Value '{{input_val_str}}' for column '{{col}}' is not a known category. Known examples: {{known_classes}}")
                                continue
                        else:
                            user_vals.append(float(val))
                            break
                    except ValueError:
                        print(f"[ERROR] Invalid number format for column '{{col}}'. Please enter a numeric value.")
                        continue
                
                if should_exit:
                    break
            
            if should_exit:
                print("\n**[INFO] Prediction mode terminated by user.**")
                break

            df_input = pd.DataFrame([user_vals], columns=feature_columns)
            user_scaled = scaler.transform(df_input) 
            
            pred_encoded = model.predict(user_scaled)[0] 
            
            pred_label = target_encoder.inverse_transform([pred_encoded])[0]
            
            print("="*30)
            print(f"**[RESULT] Predicted Class: {{pred_label}}**")
            print("="*30)

    return predict_loop

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage (Train/Predict): python script_name.py <dataset_path> <target_column> <pkl_filename>")
        print("Usage (Predict Only): python script_name.py <pkl_filename> predict_only")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    target_column = sys.argv[2]
    pkl_filename = sys.argv[3]
    
    try:
        predict_func = train_and_save_model(dataset_path, target_column, pkl_filename)
        predict_func()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {{e}}", file=sys.stderr)
        sys.exit(1)
"""

CLUSTERING_TEMPLATE = r"""
import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.cluster import KMeans # Base model for K selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier # Required for Agglomerative proxy
import warnings


DEFAULT_K_RANGE = range(3, 11) 
DEFAULT_MAX_ITER = 300
DEFAULT_N_INIT = 10 
DEFAULT_LINKAGE = "ward" 

# --- Optimal K Finding Functions ---
def find_optimal_k_generic(X_scaled, ModelClass, k_range):
    best_score = -1
    optimal_k = k_range.start
    
    print(f"[INFO] Searching for optimal K using Silhouette Score across {list(k_range)}...")
    
    for k in k_range:
        try:
            # Dynamic initialization with checks for specific required arguments
            if ModelClass.__name__ == 'AgglomerativeClustering':
                 from sklearn.cluster import AgglomerativeClustering
                 model = AgglomerativeClustering(n_clusters=k, linkage=DEFAULT_LINKAGE, metric="euclidean")
            elif ModelClass.__name__ == 'GaussianMixture':
                 from sklearn.mixture import GaussianMixture
                 model = GaussianMixture(n_components=k, random_state=42, n_init=DEFAULT_N_INIT)
            else:
                 # Standard clustering model initialization (KMeans, MiniBatchKMeans, etc.)
                 model = ModelClass(n_clusters=k, random_state=42, n_init='auto', max_iter=DEFAULT_MAX_ITER)
            
            labels = model.fit_predict(X_scaled)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    optimal_k = k
            
        except Exception:
            continue
            
    print(f"[INFO] Optimal K found: {optimal_k} (Silhouette Score: {best_score:.4f})")
    return optimal_k


def train_and_save_model(dataset_path: str, pkl_filename: str):
    warnings.filterwarnings("ignore", category=UserWarning)

    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    model = None
    scaler = None
    feature_columns = None
    encoders = {}
    optimal_k = None
    is_trained = False

    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing {{model_name}} model '{{pkl_filename}}'...**")
        try:
            model, scaler, feature_columns, encoders, optimal_k = load(full_path)
            print(f"[INFO] Model loaded successfully. Optimal K: {{optimal_k}}")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Could not load model from '{{full_path}}' due to error: {{e}}. Retraining...")
            pass

    # --- FIX APPLIED HERE: Check if dataset_path is None before attempting to load data ---
    if not is_trained and dataset_path is not None:
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
            print(f"[ERROR] Failed to load data: {{e}}", file=sys.stderr)
            sys.exit(1)

        X = data.copy()

        # --- PREPROCESSING ---
        for col in X.columns:
            if X[col].dtype == object:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
        
        X.fillna(X.mean(numeric_only=True), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_columns = list(X.columns)

        # --- OPTIMAL K SELECTION ---
        # Note: ModelClass is passed as KMeans (to simplify this template)
        # However, we must dynamically pass the suggested model class if it's Agglomerative/GMM
        from sklearn.cluster import {{model_class_name}} # Ensure the model class is imported inside the function scope for evaluation
        optimal_k = find_optimal_k_generic(X_scaled, {{model_class_name}}, DEFAULT_K_RANGE) 

        # --- TRAIN FINAL MODEL ---
        print(f"\n**[INFO] Training FINAL {{model_name}} with K={{optimal_k}}...**")
        
        # FINAL MODEL INITIALIZATION: Handle special cases for Agglomerative/KNC proxy vs native .predict()
        if 'AgglomerativeClustering' in '{{model_name}}':
            # 1. Fit AgglomerativeClustering to get the final labels
            ac_model = AgglomerativeClustering(n_clusters=optimal_k, linkage=DEFAULT_LINKAGE, metric="euclidean")
            train_labels = ac_model.fit_predict(X_scaled)
            
            # 2. Train KNC as the proxy prediction model
            model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            model.fit(X_scaled, train_labels)

        else:
            # Standard clustering model (KMeans, GMM, MiniBatchKMeans, etc.)
            model = {{model_class_name}}(n_clusters=optimal_k, random_state=42, n_init=DEFAULT_N_INIT, max_iter=DEFAULT_MAX_ITER, n_jobs=-1)
            labels = model.fit_predict(X_scaled) 
            
        # --- EVALUATION ---
        labels_to_score = train_labels if 'AgglomerativeClustering' in '{{model_name}}' else labels
        
        if optimal_k > 1 and len(np.unique(labels_to_score)) > 1:
            score = silhouette_score(X_scaled, labels_to_score)
            print(f"[INFO] Final Silhouette Score (K={{optimal_k}}): **{{score:.4f}}**")
        else:
            print("[INFO] Final model resulted in 1 cluster or fewer. Silhouette Score not calculated.")

        # --- SAVE MODEL ---
        dump((model, scaler, feature_columns, encoders, optimal_k), full_path)
        print(f"[INFO] Model saved as '{{pkl_filename}}' (K={{optimal_k}})")
        
    # --- Prediction function (Unlimited Loop) ---
    def predict_loop():
        nonlocal model, scaler, feature_columns, encoders, optimal_k
        
        if model is None:
            try:
                model, scaler, feature_columns, encoders, optimal_k = load(full_path)
            except Exception as e:
                print(f"[ERROR] Failed to load model for prediction: {{e}}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"**Starting UNLIMITED Cluster Prediction Mode for '{{pkl_filename}}'**")
        print(f"Model uses {{optimal_k}} optimized clusters.")
        print("Type 'exit' or 'quit' to close the program.")
        print("="*70)
        
        while True:
            user_vals = []
            print("\n" + "-"*30)
            
            should_exit = False
            for col in feature_columns:
                while True:
                    try:
                        val = input(f"Enter value for **{{col}}**: ")
                        input_val_str = str(val).strip()
                        
                        if input_val_str.lower() in ['exit', 'quit']:
                            should_exit = True
                            break

                        if col in encoders:
                            le = encoders[col]
                            try:
                                encoded_val = le.transform([input_val_str])[0]
                                user_vals.append(encoded_val)
                                break
                            except ValueError:
                                known_classes = list(le.classes_)
                                if len(known_classes) > 10:
                                    known_classes = known_classes[:10] + ["..."]
                                print(f"[ERROR] Value '{{input_val_str}}' for column '{{col}}' is not a known category. Known examples: {{known_classes}}")
                                continue
                        else:
                            user_vals.append(float(val))
                            break
                    except ValueError:
                        print(f"[ERROR] Invalid number format for column '{{col}}'. Please enter a numeric value.")
                        continue

                if should_exit:
                    break
            
            if should_exit:
                print("\n**[INFO] Prediction mode terminated by user.**")
                break

            df_input = pd.DataFrame([user_vals], columns=feature_columns)
            user_scaled = scaler.transform(df_input)

            cluster_id = model.predict(user_scaled)[0]
            
            print("\n" + "="*30)
            print(f"**[RESULT] Assigned Cluster: {{int(cluster_id)}}**")
            print("="*30)

    return predict_loop

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage (Train/Predict): python script_name.py <dataset_path> <pkl_filename>")
        print("Usage (Predict Only): python script_name.py <pkl_filename> predict_only")
        sys.exit(1)
        
    if len(sys.argv) == 3:
        if sys.argv[2].lower() == 'predict_only':
            pkl_filename = sys.argv[1]
            try:
                train_and_save_model(None, pkl_filename)() # Train with None path, then execute loop
            except Exception as e:
                print(f"\n[FATAL ERROR] An unexpected error occurred: {{e}}", file=sys.stderr)
                sys.exit(1)
        else:
            dataset_path = sys.argv[1]
            pkl_filename = sys.argv[2]
            try:
                train_and_save_model(dataset_path, pkl_filename)() # Train with path, then execute loop
            except Exception as e:
                print(f"\n[FATAL ERROR] An unexpected error occurred: {{e}}", file=sys.stderr)
                sys.exit(1)
    
    elif len(sys.argv) == 2:
        pkl_filename = sys.argv[1]
        try:
            train_and_save_model(None, pkl_filename)() # Transform Only Mode shortcut
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {{e}}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)
"""


def extract_filename(filepath):
    
    filename = os.path.basename(filepath)
    print(f"Filename detected: {filename}")
    return filename


def load_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)    
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    elif ext == '.json':
        df = pd.read_json(filepath)    
    elif ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif ext in ['.txt', '.log', '.md', '.conf', '.ini', '.text']:
        try:
            # Try reading as tabular data first
            df = pd.read_csv(filepath, sep=None, engine='python')
        except Exception:
            # Fallback: read as raw text
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            df = pd.read_csv(StringIO(content))
            

    elif ext in ['.tsv', '.txt']:
        df = pd.read_csv(filepath, sep="\t", header=0)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"Loaded dataset with shape {df.shape}")
    return df



def get_target_and_context(df):
    
    print("\nLet's identify your dataset details.")
    
    # Asking if dataset has a target column
    has_target = input("Does your dataset have a target column? (yes/no): ").strip().lower()
    target_column = None

    if has_target in ['yes', 'y']:
        # Askingfor the target column name
        target_column = input("Enter the target column name: ").strip()
        if target_column not in df.columns:
            print(f"Column '{target_column}' not found in dataset. Please check spelling.")
            print(f"Available columns are: {list(df.columns)}")
            confirm = input("Do you still want to proceed with this column name? (yes/no): ").strip().lower()
            if confirm not in ['yes', 'y']:
                target_column = None
    else:
        print("No target column specified — assuming this might be an unsupervised dataset.")

    # Asking for dataset context
    print("\nPlease describe what this dataset is about (for context):")
    context = input("\n ")

    # Display summary
    print("\nSummary:")
    print(f"Target Column: {target_column if target_column else 'None'}")
    print(f"Context: {context if context else 'No context provided.'} \n")

    return target_column, context

# FIX: Model and problem type are now returned as a tuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
filepath = str(REPO_ROOT / "ML_BOT" / "Sample_Datasets" / "Mall_Customers.csv")
file_name = extract_filename(filepath)
df = load_file(filepath)
target_column, context = get_target_and_context(df)

model_name, problem_type = model_suggestion(file_name, df,  target_column, context)

# --- FIX: Conditional Template Selection ---
if "regression" in problem_type.lower():
    template = REGRESSION_TEMPLATE
    # Determine model class name for dynamic injection
    model_class_name = model_name
elif "classification" in problem_type.lower():
    template = CLASSIFICATION_TEMPLATE
    model_class_name = model_name
elif "clustering" in problem_type.lower():
    template = CLUSTERING_TEMPLATE
    # FIX: For clustering, we use KMeans as the helper function input for K search.
    model_class_name = model_name 
else:
    print(f"[FATAL ERROR] Unknown problem type: {problem_type}. Cannot generate script.")
    sys.exit(1)


# FIX: Dynamically Inject Model Name, Class Name, and other variables into the final template
# This prevents the LLM from hallucinating the complex structure.
final_template = template.replace('{{model_name}}', model_name).replace('{{model_class_name}}', model_class_name)

# If clustering, we must manually inject the K-selection function, since the base model name is K-Means
if "clustering" in problem_type.lower():
    # Only need to replace ModelClass if the model is GMM or Agglomerative (KMeans is the default)
    if model_name in ['GaussianMixture', 'AgglomerativeClustering', 'Birch']:
        # We need to correctly pass the ModelClass object reference to find_optimal_k
        final_template = final_template.replace('KMeans', model_name) 

# --- SCRIPT GENERATION ---
filename = f"{model_name.replace(' ', '_')}.py"
pkl_filename = f"Trained_{model_name.replace(' ', '_')}.pkl"

prompt_template = f"""
You are an expert Python AutoML assistant and your task is to generate the final deployment-ready machine learning script.

Follow these rules strictly:

1. You MUST use the provided template EXACTLY as written.  
2. You MUST only substitute the placeholders:
   - {problem_type}
   - {model_name}
   - {model_class_name}
   - {final_template}
3. You MUST NOT modify the structure, formatting, function order, imports, CLI behavior, preprocessing logic, scaling logic, encoder handling, optimal K selection, or prediction loop.
4. The final output MUST be a complete Python script ready for execution.
5. you should never use any kind of fences, sentences, titles. just provide python code only, starting with imports and end with
last code line.

PROBLEM: {problem_type}
MODEL TO IMPLEMENT: {model_name}

TEMPLATE:
{final_template}


"""
raw_text = generate_script_content(model_name, prompt_template)
code = extract_code_block(raw_text)
save_script(filename, code)

# --- SCRIPT EXECUTION ---
# Execute script based on problem type argument structure
execute_script(filename, filepath, target_column, pkl_filename, problem_type.lower())
