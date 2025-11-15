import pandas as pd
import numpy as np
from sklearn.feature_agglomeration import FeatureAgglomeration
from sklearn.metrics import accuracy_score
from joblib import dump, load

def train_and_save_model(dataset_path, target_column, pkl_filename):
    trained_dir = "Trained_Models"
    os.makedirs(trained_dir, exist_ok=True)

    full_path = os.path.join(trained_dir, pkl_filename)

    # Load existing model
    if os.path.exists(full_path):
        print(f"Model already trained. Loading '{pkl_filename}'...")
        model, scaler, columns = load(full_path)
        print("Model loaded successfully.")
    else:
        # Load dataset
        ext = os.path.splitext(dataset_path)[1].lower()
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

        # Separate label column (may contain NaN for unlabeled data)
        y = data[target_column].fillna(-1)

        # Features
        X = data.drop(target_column, axis=1)
        X.fillna(X.mean(), inplace=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train semi-supervised model
        model = FeatureAgglomeration(n_clusters=8)
        model.fit(X_scaled, y)

        # Predict every sample (transduction)
        y_pred = model.transduction_

        # Evaluate only on known labels
        known_mask = y != -1
        if known_mask.any():
            acc = accuracy_score(y[known_mask], y_pred[known_mask])
            print(f"Training completion: Accuracy on known labels = {acc:.3f}")
        else:
            print("Training completed with no known labels (fully unsupervised mode).")

        # Save the model
        dump((model, scaler, list(X.columns)), full_path)
        print(f"Model saved as '{pkl_filename}'")

    # Prediction function
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