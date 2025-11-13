import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from joblib import dump, load
import numpy as np
import os
from io import StringIO
def train_and_save_model(dataset_path, target_column, pkl_filename):
    # Check if model already exists
    if os.path.exists(pkl_filename):
        print(f"Model already trained. Loading '{pkl_filename}'...")
        model, columns = load(pkl_filename)
        print("Model loaded successfully.")
    else:
        # Load and prepare dataset
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

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Evaluate and save
        print(f"Training completion: {model.score(X_test, y_test):.3f}")

        # Save model
        dump((model, list(X.columns)), pkl_filename)
        print(f"Model saved as '{pkl_filename}'")

    # Define the predict function (uses loaded or trained model)
    def predict():
        model, columns = load(pkl_filename)
        X_new = []
        for col in columns:
            val = float(input(f"Enter value for column {col}: "))
            X_new.append(val)
        pred = model.predict([X_new])[0]
        print(f"Predicted outcome: {pred}")

    # Return predict function so user can call it later
    return predict


# Example usage
predict_func = train_and_save_model("diabetes.csv", "Outcome", "Trained_GaussianNB.pkl")
predict_func()