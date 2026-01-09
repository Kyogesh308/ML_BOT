import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
from joblib import dump
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

def create_user_package(package_name):
    base_path = os.path.join("Trained_Models", "New_Produced_Datasets", package_name)
    os.makedirs(base_path, exist_ok=True)
    return base_path

def train_rfe(dataset_path, k_value, target_col, package_name):
    package_dir = create_user_package(package_name)
    pkl_path = os.path.join(package_dir, f"{package_name}.pkl")
    
    # 1. LOAD DATA
    data = pd.read_csv(dataset_path)
    
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

    # --- THE WRAPPER TWEAK: CHOOSE ESTIMATOR ---
    # We use Random Forest as our 'Goated' general-purpose estimator
    is_classification = data[target_col].nunique() < 20
    estimator = RandomForestClassifier(n_estimators=50, random_state=42) if is_classification else RandomForestRegressor(n_estimators=50, random_state=42)
    
    print(f"[INFO] Running RFE with {type(estimator).__name__}. Target K={k_value}...")
    
    # 2. APPLY RFE
    selector = RFE(estimator=estimator, n_features_to_select=int(k_value), step=1)
    selector.fit(X_enc, y_enc)
    
    selected_features = [col for col, keep in zip(feature_cols, selector.support_) if keep]
    removed_features = [col for col in original_cols if col not in selected_features and col != target_col]

    # --- EXPORT ---
    data[selected_features + [target_col]].to_csv(os.path.join(package_dir, "Cleaned_Dataset.csv"), index=False)
    dump((selected_features, removed_features, encoders, original_cols, target_col), pkl_path)
    
    # Save Ranking (1 is best) for Llama3 RAG
    rankings = {col: int(rank) for col, rank in zip(feature_cols, selector.ranking_)}
    with open(os.path.join(package_dir, "feature_rankings.json"), "w") as f:
        json.dump(rankings, f, indent=4)

    print(f"**[SUCCESS] RFE Complete. Saved top {k_value} features.**")

    def filter_loop():
        print("\n" + "="*70 + "\nRUNTIME FEATURE FILTER (RFE)\n" + "="*70)
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

            print(f"\n✅ RETAIN: {selected_features + [target_col]}")
            print(f"❌ ELIMINATE: {removed_features}")
            filtered = [val for col, val in zip(original_cols, user_input) if col in selected_features or col == target_col]
            print(f"Filtered Row: {filtered}\n" + "-"*42)

    return filter_loop

if __name__ == "__main__":
    if len(sys.argv) == 5:
        # Usage: python RFE.py data.csv 3 "Target" my_rfe_pkg
        run = train_rfe(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        if run: run()