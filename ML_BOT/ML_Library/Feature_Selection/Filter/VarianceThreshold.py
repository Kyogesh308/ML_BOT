import pandas as pd
import numpy as np
import os
import sys
import warnings
from joblib import dump
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def train_variance_selector(dataset_path, raw_variance_threshold, package_name):
    package_dir = os.path.join("Trained_Models", "New_Produced_Datasets", package_name)
    os.makedirs(package_dir, exist_ok=True)
    pkl_path = os.path.join(package_dir, f"{package_name}.pkl")
    
    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"[ERROR] Loading failed: {e}"); return None

    # --- ARCHITECT STEP: ID PROTECTION ---
    # Automatically identify and drop potential ID columns from the SELECTION process
    # We don't want to calculate variance for something like 'CustomerID'
    potential_ids = [col for col in data.columns if 'id' in col.lower() or 'index' in col.lower()]
    print(f"[ADVICE] Detected potential ID columns: {potential_ids}. These will be removed automatically.")
    
    # Work on a copy for selection
    df_selection = data.drop(columns=potential_ids)
    original_cols = data.columns.tolist()
    selection_cols = df_selection.columns.tolist()
    
    # --- PREPROCESS ---
    df_temp = df_selection.copy()
    encoders = {}
    for col in df_temp.columns:
        if df_temp[col].dtype == object:
            le = LabelEncoder()
            df_temp[col] = le.fit_transform(df_temp[col].astype(str))
            encoders[col] = le
    
    df_temp.fillna(df_temp.mean(numeric_only=True), inplace=True)

    # --- APPLY SELECTOR ---
    print(f"[INFO] Llama3 Threshold: {raw_variance_threshold}")
    selector = VarianceThreshold(threshold=raw_variance_threshold)
    selector.fit(df_temp)
    
    # Support tells us which of the SELECTION columns survived
    selected_mask = selector.get_support()
    selected_features = [col for col, keep in zip(selection_cols, selected_mask) if keep]
    
    # Final List = Selected Features (ID columns are gone forever)
    removed_features = [col for col in original_cols if col not in selected_features]

    # --- EXPORT ---
    data[selected_features].to_csv(os.path.join(package_dir, "Cleaned_Dataset.csv"), index=False)
    dump((selected_features, removed_features, encoders, original_cols), pkl_path)
    
    # Save a detailed report for Llama3 RAG
    with open(os.path.join(package_dir, "dataset_stats.json"), "w") as f:
        import json
        json.dump({col: float(v) for col, v in zip(selection_cols, selector.variances_)}, f)

    print(f"**[SUCCESS] Cleaned dataset excludes IDs and low-variance features.**")

    def filter_loop():
        print("\n" + "="*70 + "\nRUNTIME FEATURE FILTER\n" + "="*70)
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

            print(f"\n✅ RETAIN: {selected_features}")
            print(f"❌ ELIMINATE: {removed_features}")
            filtered = [val for col, val in zip(original_cols, user_input) if col in selected_features]
            print(f"Filtered Vector: {filtered}\n" + "-"*42)

    return filter_loop

if __name__ == "__main__":
    if len(sys.argv) == 4:
        run = train_variance_selector(sys.argv[1], float(sys.argv[2]), sys.argv[3])
        if run: run()