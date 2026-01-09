import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
from joblib import dump
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def create_user_package(package_name):
    base_path = os.path.join("Trained_Models", "New_Produced_Datasets", package_name)
    os.makedirs(base_path, exist_ok=True)
    return base_path

def train_select_k_best(dataset_path, k_value, target_col, package_name):
    """
    k_value: Number of top features to keep (provided by Llama3).
    target_col: The column we are trying to predict.
    """
    package_dir = create_user_package(package_name)
    pkl_path = os.path.join(package_dir, f"{package_name}.pkl")
    
    # 1. LOAD DATA
    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"[ERROR] Loading failed: {e}"); return None

    if target_col not in data.columns:
        print(f"[ERROR] Target column '{target_col}' not found."); return None

    # --- ARCHITECT STEP: ID PROTECTION ---
    potential_ids = [col for col in data.columns if 'id' in col.lower() or 'index' in col.lower()]
    
    # Separate Features (X) and Target (y)
    X_raw = data.drop(columns=[target_col] + potential_ids)
    y_raw = data[target_col]
    original_cols = data.columns.tolist()
    feature_cols = X_raw.columns.tolist()

    # --- PREPROCESS ---
    X_enc = X_raw.copy()
    encoders = {}
    for col in X_enc.columns:
        if X_enc[col].dtype == object:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col].astype(str))
            encoders[col] = le
    
    X_enc.fillna(X_enc.mean(numeric_only=True), inplace=True)

    # Encode target if it's categorical
    y_enc = y_raw
    if y_raw.dtype == object:
        le_y = LabelEncoder()
        y_enc = le_y.fit_transform(y_raw.astype(str))
        encoders[target_col] = le_y

    # --- APPLY SELECTKBEST ---
    # Choose test based on target type (Classification vs Regression)
    # We'll use f_classif by default, or f_regression if target is continuous
    score_func = f_classif if data[target_col].nunique() < 20 else f_regression
    
    print(f"[INFO] Running SelectKBest (K={k_value}) using {score_func.__name__}...")
    
    # Handle K="all" or K > total features
    k_val = min(int(k_value), len(feature_cols))
    
    selector = SelectKBest(score_func=score_func, k=k_val)
    selector.fit(X_enc, y_enc)
    
    selected_mask = selector.get_support()
    selected_features = [col for col, keep in zip(feature_cols, selected_mask) if keep]
    removed_features = [col for col in original_cols if col not in selected_features and col != target_col]

    # --- EXPORT PACKAGE ---
    # The new dataset keeps the selected features AND the target column
    cleaned_df = data[selected_features + [target_col]]
    cleaned_df.to_csv(os.path.join(package_dir, "Cleaned_Dataset.csv"), index=False)
    
    dump((selected_features, removed_features, encoders, original_cols, target_col), pkl_path)
    
    # Save scores for Llama3 RAG
    feature_scores = {col: float(s) for col, s in zip(feature_cols, selector.scores_)}
    with open(os.path.join(package_dir, "feature_importance_scores.json"), "w") as f:
        json.dump(feature_scores, f, indent=4)

    print(f"**[SUCCESS] Package created. Top {k_val} features selected based on target impact.**")

    # --- FILTER LOOP ---
    def filter_loop():
        print("\n" + "="*70 + "\nRUNTIME FEATURE FILTER (SelectKBest)\n" + "="*70)
        while True:
            user_input = []
            should_exit = False
            for col in original_cols:
                while True:
                    val = input(f"Enter {col}: ").strip()
                    if val.lower() in ['exit', 'quit']: should_exit = True; break
                    if not val: continue
                    user_input.append(val); break
                if should_exit: break
            if should_exit: break

            print(f"\n✅ RETAIN: {selected_features + [target_col]}")
            print(f"❌ ELIMINATE: {removed_features}")
            filtered = [val for col, val in zip(original_cols, user_input) if col in selected_features or col == target_col]
            print(f"Filtered Vector: {filtered}\n" + "-"*42)

    return filter_loop

if __name__ == "__main__":
    if len(sys.argv) == 5:
        # Usage: python SelectKBest.py Mall_Customers.csv 3 "Spending Score (1-100)" my_kbest_pkg
        run = train_select_k_best(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        if run: run()
    else:
        print("Usage: python SelectKBest.py <dataset.csv> <K> <target_column> <package_name>")