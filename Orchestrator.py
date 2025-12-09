import os
from pathlib import Path
import pandas as pd
import string

from ModelSuggestion import model_suggestion
from ModelGenerator import *
from io import StringIO
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
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "ML_BOT" / "Sample_Datasets" / "Mall_Customers.csv"

filepath = str(DEFAULT_DATASET)
file_name = extract_filename(filepath)
#loading the file into the dataframe
df = load_file(filepath)

target_column, context = get_target_and_context(df)



model = model_suggestion(file_name, df,  target_column, context)

column_names = df.columns.tolist()


filename = f"{model.replace(' ', '_')}.py"

prompt_template = rf"""
You are an expert Python AutoML assistant.

Generate a complete, clean, and executable Python script for a supervised learning task using scikit-learn.
Do not include any explanations or markdown outside the code.  
Output only the Python code.

---

USER INPUTS
- Dataset path: `{filepath}`
- Target column: `{target_column}`
- Model name: `{model}` (from scikit-learn)

---

SCRIPT REQUIREMENTS

1.Import necessary libraries
   - pandas, scikit-learn (train_test_split, model import), joblib, numpy.

2. Load dataset
   - Load the dataset from `{filepath}` using pandas.
   - Separate the target column `{target_column}` as y.
   - Use the remaining columns as X.

3.Preprocessing
   - Handle any missing values by filling with column mean.
   - Standardize features using StandardScaler.

4.Train-test split
   - Perform an 80/20 train-test split (random_state=42).

5.Model training
   - Initialize and train the model `{model}` using X_train and y_train.

6.Evaluation
   - Make predictions on X_test.
   - Print training completion and evaluation results (accuracy or R² based on task type).

7. Model saving
   - Save the trained model using joblib as `"Trained_{model}.pkl"`.

8. Prediction prompt
   - Prompt the user to input new feature values (one by one for each column in X).
   - Predict the target using the trained model.
   - Print the prediction clearly.

---

 OUTPUT FORMAT
Output only the complete Python code, enclosed in:
```python
 (the script)```
 
 the example script if you want to follow: 
 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
import numpy as np
import os

data = pd.read_csv(REPO_ROOT / 'ML_BOT' / 'Sample_Datasets' / 'diabetes.csv')
y = data['Outcome']
X = data.drop('Outcome', axis=1)

X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Training completion: model.score(X_test, y_test)')

dump_dir = 'models'
dump_filename = 'Trained_Decision_Tree_Classifier.pkl'

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

dump(model, os.path.join(dump_dir, dump_filename))

while True:
    features = []
    for col in X.columns:
        val = float(input(f'Enter value for column col: '))
        features.append(val)
    features = np.array([features]).reshape(1, -1)  # Add this line
    pred = model.predict(features)
    print(f'Predicted outcome: pred[0]')

"""
raw_text = generate_script_content(model, prompt_template)
code = extract_code_block(raw_text)
save_script(filename, code)
execute_script(filename, filepath, target_column, f"Trained_{model}.pkl")

