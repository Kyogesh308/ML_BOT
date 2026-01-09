import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
from joblib import dump
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold

# Silence scikit-learn chatter
warnings.filterwarnings("ignore", category=UserWarning)

def create_user_package(package_name):
    base_path = os.path.join("Trained_Models", "New_Produced_Datasets", package_name)
    os.makedirs(base_path, exist_ok=True)
    return base_path

def train_rfecv(dataset_path, target_col, package_name):
    package_dir = create_user_package(package_name)
    pkl_path = os.path.join(package_dir, f"{package_name}.pkl")
    
    # 1. LOAD DATA
    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"[ERROR] Load failed: {e}"); return None

    # --- ARCHITECT STEP: ID PROTECTION ---
    potential_ids = [col for col in data.columns if 'id' in col.lower() or 'serial' in col.lower()]
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

    y_enc = y_raw
    if y_raw.dtype == object:
        le_y = LabelEncoder()
        y_enc = le_y.fit_transform(y_raw.astype(str).str.strip())
        encoders[target_col] = le_y

    # --- CHOOSE ESTIMATOR & CV STRATEGY ---
    is_classification = data[target_col].nunique() < 20
    if is_classification:
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        cv = StratifiedKFold(5)
        scoring = "accuracy"
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        cv = KFold(5)
        scoring = "neg_mean_squared_error"
    
    print(f"[INFO] Running RFECV. Searching for the OPTIMAL number of features...")

    # 2. APPLY RFECV
    selector = RFECV(estimator=estimator, step=1, cv=cv, scoring=scoring, n_jobs=-1)
    selector.fit(X_enc, y_enc)
    
    selected_features = [col for col, keep in zip(feature_cols, selector.support_) if keep]
    removed_features = [col for col in original_cols if col not in selected_features and col != target_col]

    # --- EXPORT PACKAGE ---
    data[selected_features + [target_col]].to_csv(os.path.join(package_dir, "Cleaned_Dataset.csv"), index=False)
    dump((selected_features, removed_features, encoders, original_cols, target_col), pkl_path)
    
    # Save the CV results for Llama3 RAG
    report = {
        "optimal_feature_count": int(selector.n_features_),
        "rankings": {col: int(rank) for col, rank in zip(feature_cols, selector.ranking_)},
        "cv_scores": selector.cv_results_['mean_test_score'].tolist()
    }
    with open(os.path.join(package_dir, "rfecv_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print(f"**[SUCCESS] RFECV found {selector.n_features_} optimal features.**")

    # --- FILTER LOOP ---
    def filter_loop():
        print("\n" + "="*70 + "\nRUNTIME FEATURE FILTER (RFECV)\n" + "="*70)
        while True:
            user_input, should_exit = [], False
            for col in original_cols:
                while True:
                    val = input(f"Enter {col}: ").strip()
                    if val.lower() in ['exit', 'quit']: should_exit = True; break
                    if not val: continue
                    user_input.append(val); break
                if should_exit: break
            if should_exit: break

            print(f"\n✅ RETAIN (Optimal Subset): {selected_features + [target_col]}")
            print(f"❌ ELIMINATE: {removed_features}")
            filtered = [val for col, val in zip(original_cols, user_input) if col in selected_features or col == target_col]
            print(f"Filtered Row: {filtered}\n" + "-"*42)

    return filter_loop

if __name__ == "__main__":
    if len(sys.argv) == 4:
        # Usage: python RFECV.py data.csv "Target" my_package
        run = train_rfecv(sys.argv[1], sys.argv[2], sys.argv[3])
        if run: run()