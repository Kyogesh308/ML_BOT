import os
import json
import pandas as pd
import subprocess
import sys
from io import StringIO
from pathlib import Path
import shutil
from generate_package import create_user_package
# --- GLOBAL CONFIGURATION AND SCHEMA ---

# Your specific paths
ML_LIBRARY_ROOT = r"C:\Users\konda\OneDrive\Desktop\projects\ML_BOT\ML_BOT\ML_Library"
TRAINED_MODELS_DIR = r"C:\Users\konda\OneDrive\Desktop\projects\ML_BOT\Trained_Models"
HARDCODED_DATASET_PATH = r"C:\Users\konda\OneDrive\Desktop\projects\ML_BOT\ML_BOT\Sample_Datasets\height-weight.csv" 

# Model Schema: Mapping problem types to EXACT CamelCase script names
MODEL_SCHEMA = {
    "Classification": [
        "LogisticRegression", "KNearestNeighbors", "DecisionTreeClassifier", 
        "RandomForestClassifier", "GradientBoostingClassifier", 
        "SupportVectorMachine", "NaiveBayes"
    ],
    "Regression": [
        "LinearRegression", "RidgeRegression", "LassoRegression", 
        "ElasticNet", "KNearestNeighborsRegressor", "DecisionTreeRegressor", 
        "RandomForestRegressor"
    ],
    "Clustering": [
        "KMeans", "DBSCAN", "AgglomerativeClustering", "MeanShift"
    ],
    "DimensionalityReduction": [
        "PCA", "TSNE", "LDA", "KernelPCA"
    ]
}

# --- 2. DATA UTILITY FUNCTIONS ---

def read_dataset(filepath: str) -> pd.DataFrame | None: 
    """Reads the dataset based on file extension (csv, xlsx, json, txt, tsv)."""
    try:
        file_extension = Path(filepath).suffix.lower()
        if file_extension == '.csv': df = pd.read_csv(filepath)
        elif file_extension in ('.xlsx', '.xls'): df = pd.read_excel(filepath)
        elif file_extension == '.json': df = pd.read_json(filepath)
        elif file_extension == '.txt' or file_extension == '.tsv':
            # Assuming space or tab-separated for txt/tsv
            df = pd.read_csv(filepath, sep=r'\s+|,', engine='python')
        else:
            print(f"Unsupported file type: {file_extension}")
            return None
        
        print(f"✅ Dataset loaded successfully: {Path(filepath).name}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def generate_stats(df: pd.DataFrame) -> str:
    """Calculates comprehensive dataset statistics and returns them as a single string. Requires 'tabulate'."""
    stats_output = StringIO()
    df.info(buf=stats_output)
    stats_output.write("\n--- Descriptive Statistics (All Columns) ---\n")
    stats_output.write(df.describe(include='all').to_markdown())
    stats_output.write("\n\n--- Missing Value Counts ---\n")
    stats_output.write(df.isnull().sum().to_markdown(numalign="left", stralign="left"))
    return stats_output.getvalue()

# --- 3. LLM INTERACTION (Using Ollama) ---

def get_model_suggestion_from_llama3(stats: str, filename: str, target_col: str, context: str) -> dict | None: 
    """
    Calls Llama 3 via Ollama, constraining the 'suggested_model' field to CamelCase names.
    """
    try:
        import ollama
    except ImportError:
        print("❌ Ollama library not found. Please run 'pip install ollama'.")
        return None
    
    # Generate the model lists for the prompt
    classification_models = ', '.join(MODEL_SCHEMA['Classification'])
    regression_models = ', '.join(MODEL_SCHEMA['Regression'])
    clustering_models = ', '.join(MODEL_SCHEMA['Clustering'])
    dr_models = ', '.join(MODEL_SCHEMA['DimensionalityReduction'])
    
    # Define the required JSON schema as a string for the prompt
    json_schema = """
    {
        "problem_type": "Classification" | "Regression" | "Clustering" | "DimensionalityReduction",
        "suggested_model": "string",
        "reasoning": "string",
        "unnecesarry_column_in_the_dataset": "string"
    }
    """
    
    # --- UPDATED PROMPT: Constrains the suggested_model field ---
    prompt = f"""
    You are an expert Machine Learning assistant. Analyze the following dataset information and provide an assessment.

    - **Filename:** {filename}
    - **User Context/Goal:** {context}
    - **Target Column:** {target_col if target_col else 'None provided'}
    - **Dataset Statistics:**
    {stats}

    Based on the analysis, determine the 'problem_type' and suggest the best model.
    
    CRITICAL INSTRUCTION: The 'suggested_model' MUST be chosen EXACTLY from the following list.
    - Classification: {classification_models}
    - Regression: {regression_models}
    - Clustering: {clustering_models}
    - DimensionalityReduction: {dr_models}

    Your response MUST be ONLY a single JSON object that strictly follows this schema.
    
    JSON SCHEMA:
    {json_schema}
    """
    # --- END OF UPDATED PROMPT ---
    
    print("\n--- 🤖 Sending Analysis to Llama 3 via Ollama... ---")
    
    try:
        response = ollama.generate(
            model='llama3', 
            prompt=prompt,
            format="json",
            options={'temperature': 0.0}
        )
        
        json_str = response['response']
        print(f"✅ Ollama response received.")
        
        return json.loads(json_str)

    except ollama.OllamaAPIError as e:
        print("❌ Ollama API Error: Ensure 'llama3' is pulled (ollama pull llama3) and the server is running.")
        print(e)
        return None
    except json.JSONDecodeError:
        print(f"❌ JSON Decode Error: Llama 3 did not return valid JSON. Check the raw output for errors.")
        print(f"Raw response (first 200 chars): {json_str[:200]}...")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during LLM call: {e}")
        return None

# --- 4. MODEL SEARCH FUNCTION ---

def find_model_script(library_path, problem_type, script_filename: str) -> str | None: 
    """
    Finds the model script in the ML_Library using problem_type and the model filename.
    """
    problem_dir_path = os.path.join(library_path, problem_type)

    if not os.path.isdir(problem_dir_path):
        print(f"Error: Problem directory '{problem_dir_path}' not found in library.")
        return None

    for root, _, files in os.walk(problem_dir_path):
        if script_filename in files:
            model_script_path = os.path.join(root, script_filename)
            print(f"✅ Found script for {problem_type} at: {model_script_path}")
            return model_script_path
    
    print(f"❌ Could not find '{script_filename}' in any subdirectory of '{problem_type}'.")
    return None

# --- 5. MODEL EXECUTION (Using Subprocess) ---

import subprocess 
import sys 
from pathlib import Path
import os # Ensure os is imported for os.devnull if needed, though stdout=None usually suffices

def execute_model_script(script_path: str, data_filepath: str, target_col: str | None, model_output_dir: str):
    """
    Executes the model script using subprocess.run with inherited I/O.
    This allows the child script to run its training and then enter its 
    interactive 'unlimited prediction' loop, waiting for direct keyboard input 
    from the user until the user types the exit command.
    """
    
    # 1. Setup and Path Generation
    model_script_name = Path(script_path).stem
    pkl_filename_full = f"{model_script_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    pkl_full_path = os.path.join(model_output_dir, pkl_filename_full)
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n--- 🧠 Executing Model Training Script (Interactive Execution) ---")
    
    # 2. Build the Command (Positional Arguments)
    command = [
        sys.executable, 
        script_path, 
        data_filepath,  # 1st POSITIONAL ARGUMENT: <dataset_path>
    ]
    
    if target_col:
        # Supervised: <dataset_path> <target_column> <pkl_filename>
        print(f"-> Passing supervised arguments: {data_filepath}, {target_col}, {pkl_full_path}")
        command.extend([
            target_col,     # 2nd POSITIONAL ARGUMENT: <target_column>
            pkl_full_path   # 3rd POSITIONAL ARGUMENT: <pkl_filename>
        ])
    else:
        # Unsupervised: <dataset_path> <pkl_filename>
        print(f"-> Passing unsupervised arguments: {data_filepath}, {pkl_full_path}")
        command.extend([
            pkl_full_path   # 2nd POSITIONAL ARGUMENT: <pkl_filename>
        ])
    
    print(f"Command: {' '.join(command)}")

    # 3. Execute the command and allow full user interaction
    try:
        # Use subprocess.run and let stdout/stderr/stdin be inherited (the default).
        # We explicitly set check=True to raise an error if the process fails.
        result = subprocess.run(
            command,
            check=True,
            # IMPORTANT: We do NOT use capture_output=True or pipe stdin/stdout.
            # This ensures the child process's I/O is directed to the user's terminal.
            text=True 
        )
        
        # If the script finishes without error, it means the user manually exited the prediction loop.
        
        print("\n✅ Model script exited successfully after user input.")
        print(f"Model saved to: {pkl_full_path}")
        
    except subprocess.CalledProcessError as e:
        # This catches errors if the child script crashes during training or prediction.
        print(f"❌ Error during script execution (Return Code {e.returncode})")
        print("Please check the terminal output for the script's internal error messages.")
    except FileNotFoundError:
        print("❌ Error: Python executable or script not found. Check paths.")
    return pkl_full_path
        
# --- 6. MAIN ORCHESTRATOR FUNCTION ---

def orchestrate_ml_pipeline():
    """Orchestrates the entire ML discovery and training pipeline."""
    
    # 1. Interactive Input
    print("="*50)
    print("✨ ML Model Orchestration Pipeline ✨")
    print(f"Dataset Filepath: {HARDCODED_DATASET_PATH}")
    print("="*50)
    
    target_column = input(f"Enter the TARGET COLUMN name (or leave blank if unsupervised): ").strip() or None
    user_context = input("Enter your CONTEXT/GOAL for this dataset (e.g., 'Predict survival'): ").strip()
    if not user_context:
        user_context = "Analyze the data and suggest a fitting ML task."

    dataset_path = HARDCODED_DATASET_PATH
    
    print("\n--- Starting Data Analysis ---")
    
    df = read_dataset(dataset_path)
    if df is None:
        return

    file_name = Path(dataset_path).name
    stats_string = generate_stats(df)
    print("The Calculated stats are: ", stats_string)
    
    llm_result = get_model_suggestion_from_llama3(stats_string, file_name, target_column, user_context)
    
    if llm_result is None:
        print("Pipeline aborted due to LLM error.")
        return

    problem_type = llm_result.get('problem_type')
    suggested_model = llm_result.get('suggested_model')
    
    print("\n" + "="*50)
    print(f"🤖 LLM Problem Assessment")
    print("="*50)
    print(f"  Problem Type: **{problem_type}**")
    print(f"  Suggested Model: **{suggested_model}**")
    print(f"  Reasoning: {llm_result.get('reasoning')}")
    print(f"  Unnecessary Columns: {llm_result.get('unnecesarry_column_in_the_dataset')}")
    print("="*50 + "\n")

    # 6. Find the model script (Uses the exact name from LLM)
    model_script_name = f"{suggested_model}.py"
    
    script_path = find_model_script(
        library_path=ML_LIBRARY_ROOT,
        problem_type=problem_type,
        script_filename=model_script_name
    )

    if script_path:
        # 7. Train the model
        pklpath = execute_model_script(
            script_path=script_path,
            data_filepath=dataset_path,
            target_col=target_column,
            model_output_dir=TRAINED_MODELS_DIR
        )
    
    answer = input("Do You want an app to download the trained model? (yes or no): ")
    if 'y' in answer.lower():
        appName = 'ML_App'
        appName = input("Enter the name for your App of your own choice: ")
        if 'Run_App' not in appName:
            print(f"Your App named '{appName}' is being build...")
        else:
            print(f"Your App with name 'ML_App' is being build...")
        create_user_package(appName, pklpath, problem_type)
    
    for item in os.listdir("Trained_Models"):
        path = os.path.join("Trained_Models", item)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    print("\n All the trained models are deleted!!")

        
    print("\n✨ ML Orchestration Pipeline Complete. Check the Trained_Models directory! ✨")


# --- 7. EXECUTION BLOCK (Final, Corrected) ---

if __name__ == "__main__":
    # Check 1: Ollama
    try:
        import ollama
    except ImportError:
        print("FATAL ERROR: The 'ollama' library is not installed.")
        print("Please install it using: pip install ollama")
        sys.exit(1)
        
    # Check 2: tabulate (Corrected check for 'tabulate' dependency)
    try:
        # Check using a minimal, non-empty DataFrame to avoid the "Cannot describe" ValueError
        pd.DataFrame({'A': [1]}).describe().to_markdown() 
    except ImportError:
        print("FATAL ERROR: The 'tabulate' library is missing (required by pandas.to_markdown()).")
        print("Please install it using: pip install tabulate")
        sys.exit(1)

    orchestrate_ml_pipeline()