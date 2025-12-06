import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer 

# --- Constants ---
DEFAULT_TEST_SIZE = 0.2

def train_and_save_model(dataset_path: str, target_column: str, pkl_filename: str):
    """
    Trains a TransformedTargetRegressor model (using LinearRegression) 
    with robust preprocessing pipeline and saves the model.
    """
    
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)
    full_path = os.path.join(trained_dir, pkl_filename)

    # Variables for saving/loading
    model = None 
    feature_columns = None
    is_trained = False

    # --- CASE 1: Load existing model ---
    if os.path.exists(full_path):
        print(f"**[INFO] Loading existing Transformed Target Regressor model '{pkl_filename}'...**")
        try:
            # Load clean two-tuple: model, feature_columns
            model, feature_columns = load(full_path) 
            print("[INFO] Model loaded successfully.")
            is_trained = True
        except Exception as e:
            print(f"[WARNING] Could not load model from '{full_path}' due to error: {e}. Retrying with training...")
            pass


    # --- CASE 2: Train a new model ---
    if not is_trained:
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
            
        if target_column not in data.columns:
            raise KeyError(f"Target column '{target_column}' not found in the dataset.")

        y = data[target_column]
        X = data.drop(target_column, axis=1)
        
        # --- IDENTIFY COLUMNS ---
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
        feature_columns = X.columns.tolist() # Preserve the original order/names for input prompt consistency

        # --- CREATE PREPROCESSING PIPELINE (ColumnTransformer) ---
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop' 
        )

        # --- BASE REGRESSOR (Linear Regression inside Preprocessor Pipeline) ---
        base_regressor = Pipeline(steps=[
            ('preprocessor', preprocessor), 
            ('regressor', LinearRegression(n_jobs=-1))
        ])
        
        # --- TRANSFORMED TARGET REGRESSOR ---
        print("\n**[INFO] Building and Training Transformed Target Regressor...**")
        
        model = TransformedTargetRegressor(
            regressor=base_regressor,
            transformer=QuantileTransformer(output_distribution='normal', n_quantiles=1000, subsample=20000) 
        )
        
        # --- SPLIT DATA ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=DEFAULT_TEST_SIZE, random_state=42
        )
        
        # --- FIT MODEL ---
        model.fit(X_train, y_train)

        # --- EVALUATION ---
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"\n[INFO] R-squared Score (Test): **{r2:.4f}**")
        print(f"[INFO] Mean Squared Error (Test): **{mse:.4f}**")

        # --- SAVE MODEL ---
        # Save the model and the feature column list
        dump((model, feature_columns), full_path)
        print(f"[INFO] Model saved as '{pkl_filename}'")
        is_trained = True

    # --- Prediction function (Unlimited Loop) ---
    def predict_loop():
        nonlocal model, feature_columns
        
        if model is None:
            try:
                model, feature_columns = load(full_path)
            except Exception as e:
                print(f"[ERROR] Failed to load model for prediction: {e}", file=sys.stderr)
                return

        print("\n" + "="*70)
        print(f"**Starting UNLIMITED Prediction Mode for '{pkl_filename}'**")
        print("Type 'exit' or 'quit' to close the program.")
        print("="*70)
        
        while True:
            user_input_dict = {}
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

                        # Determine if input is meant to be numeric or categorical based on parsing attempt
                        try:
                            user_input_dict[col] = float(input_val_str)
                        except ValueError:
                            user_input_dict[col] = input_val_str
                        break
                        
                    except Exception as e:
                        print(f"[ERROR] Invalid input: {e}")
                        continue
                
                if should_exit:
                    break
            
            if should_exit:
                print("\n**[INFO] Prediction mode terminated by user.**")
                break

            # Convert user input to a DataFrame with correct columns and dtypes
            # The saved model's pipeline handles all imputation, encoding, and scaling automatically.
            df_input = pd.DataFrame([user_input_dict], columns=feature_columns)
            
            pred = model.predict(df_input)[0]
            
            print("\n" + "="*30)
            print(f"**[RESULT] Predicted Continuous Outcome: {pred:.4f}**")
            print("="*30)

    return predict_loop

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage (Train/Predict): python script_name.py <dataset_path> <target_column> <pkl_filename>")
        print("Usage (Predict Only): python script_name.py <pkl_filename> predict_only")
        sys.exit(1)
        
    if len(sys.argv) == 4:
        # Standard Training and Prediction
        dataset_path = sys.argv[1]
        target_column = sys.argv[2]
        pkl_filename = sys.argv[3]
        
        try:
            predict_func = train_and_save_model(dataset_path, target_column, pkl_filename)
            predict_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif len(sys.argv) == 3 and sys.argv[2].lower() == 'predict_only':
        # Prediction Only Mode
        pkl_filename = sys.argv[1]
        
        try:
            # Use dummy None values for path/target column
            predict_func = train_and_save_model(None, None, pkl_filename) 
            predict_func()
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("\n[FATAL ERROR] Invalid number of arguments provided.")
        sys.exit(1)