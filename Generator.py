
import ollama
import sys

OLLAMA_MODEL = "llama3"


def generate_script_content(model: str, prompt_template: str) -> str:
    """
    Calls Ollama to generate a Python script based on the provided model and prompt template.
    """
    print(f"1. Calling Ollama client with model '{model}' to generate script...")

    refined_prompt = f"ONLY output the Python code, with no explanations, markdown, or extra text. {prompt_template}, for this model: {model}"

    try:
        client = ollama.Client()
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=refined_prompt,
            options={'temperature': 0.0}
        )
        raw_text = response['response'].strip()
        print("   Ollama output received via Python client.")
        return raw_text
    except ollama.OllamaAPIError as e:
        print(f"\nError: Ollama API call failed. Is Ollama running and is the model '{model}' downloaded?")
        print(f"  Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during Ollama generation: {e}")
        sys.exit(1)

def extract_code_block(raw_text: str) -> str:
    """
    Strips markdown fences from Ollama's output and returns clean Python code.
    """
    
    code_content = raw_text.strip()
    original_content = code_content
    fences = ['```python', '```py', '```']

    for fence in fences:
        if code_content.lower().startswith(fence):
            code_content = code_content[len(fence):].lstrip()
            break

    if code_content.rstrip().endswith('```'):
        last_fence_index = code_content.rfind('```')
        if last_fence_index != -1:
            code_content = code_content[:last_fence_index].rstrip()

    code_content = code_content.strip()

    if code_content != original_content:
        print("2. Extracted and cleaned code by aggressively stripping markdown fences.")
    elif original_content.startswith('```') or original_content.endswith('```'):
        print("2. Warning: Markdown fences detected but stripping failed to change content.")
    else:
        print("2. Assuming raw output is the script content (no fences found/removed).")

    return code_content


import os

def save_script(filename: str, content: str) -> None:
    """
    Saves the given Python script content into:
    ML_BOT/ML_Library/Classification/Linear_models/
    Creates the directory if it doesn't exist.
    """
    # Define the base save directory
    save_dir = os.path.join("ML_BOT", "ML_Library", "Regression", "Linear_Models")
    
    # Make sure the full directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the full file path
    file_path = os.path.join(save_dir, filename)
    
    print(f"3. Saving code to file: {file_path}")
    
    # Save the content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ File saved successfully at: {file_path}")

if __name__ == "__main__":
    regression_models = [
    # Linear Models
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "Lars",
    "LassoLars",
    "OrthogonalMatchingPursuit",
    "BayesianRidge",
    "ARDRegression",
    "HuberRegressor",
    "TheilSenRegressor",
    "RANSACRegressor",
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
    "QuantileRegressor",
    "PassiveAggressiveRegressor",
    "SGDRegressor",]
    
    print("Generating and saving Linear_Models. ")
    prompt_template = """
You are an expert AutoML Python code generator.

Your goal: Generate a **complete, runnable Python script** for a regression model using scikit-learn.

---

### INPUT:
You will receive one model name (e.g. "LinearRegression", "RandomForestRegressor", etc.).

All models belong to the **regression family** from scikit-learn, so import them correctly from their submodules such as:
- sklearn.linear_model
- sklearn.ensemble
- sklearn.svm
- sklearn.tree
- sklearn.naive_bayes
- sklearn.neighbors
- sklearn.discriminant_analysis

---

### OUTPUT:
Generate a fully functional Python script that exactly follows the structure and code conventions below.

---

### REFERENCE TEMPLATE (REPRODUCE THIS EXACT STRUCTURE)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import numpy as np
import os
from io import StringIO
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

        model = LogisticRegression()
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
"""

    for i in regression_models:
        raw = generate_script_content(i, prompt_template)
        content = extract_code_block(raw)
        filename = f"{i.replace(' ', '_')}.py"
        save_script(filename, content)
        print(i, "script created, going on to the next")
        
        