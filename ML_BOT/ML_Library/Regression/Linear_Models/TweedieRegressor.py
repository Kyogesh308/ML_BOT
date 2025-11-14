import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import TweedieRegressor
from joblib import dump, load
import numpy as np
import os
import sys
def train_and_save_model(dataset_path, target_column, pkl_filename):
    trained_dir = "Trained_Models"
    full_path = os.path.join(trained_dir, pkl_filename)
    if os.path.exists(full_path):
        print(f"Model already trained. Loading '{pkl_filename}'...")
        model, scaler, columns = load(full_path)
        print("Model loaded successfully.")
    else:
        ext = os.path.splitext(dataset_path)[1].lower()
        if ext == '.csv':
            data = pd.read_csv(dataset_path)    
        elif ext in ['.xls', '.xlsx']:
            data = pd.read_excel(dataset_path)
        elif ext == '.json':
            data = pd.read_json(dataset_path)    
        elif ext == '.parquet':
            data = pd.read_parquet(dataset_path)
        elif ext in ['.txt', '.log', '.md', '.conf', '.ini', '.text']:
            try:
                # Try reading as tabular data first
                data = pd.read_csv(dataset_path, sep=None, engine='python')
            except Exception:
                # Fallback: read as raw text
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                data = pd.read_csv(StringIO(content))
        
        y = data[target_column]
        X = data.drop(target_column, axis=1)

        X.fillna(X.mean(), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = TweedieRegressor()
        model.fit(X_train, y_train)

        print(f"Training score: {model.score(X_test, y_test):.3f}")

        dump((model, scaler, list(X.columns)), full_path)
        print(f"Model saved as '{pkl_filename}'")

    def predict():
        model, scaler, columns = load(full_path)
        X_new = []
        for col in columns:
            val = float(input(f"Enter value for column {col}: "))
            X_new.append(val)
        X_new = scaler.transform([X_new])
        pred = model.predict(X_new)[0]
        print(f"Predicted outcome: {pred}")

    return predict

if __name__ == "__main__":
    predict_func = train_and_save_model(sys.argv[1], sys.argv[2], sys.argv[3])
    predict_func()