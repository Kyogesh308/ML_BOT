import os
import json
import pandas as pd
import subprocess
import sys
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values

# --- GLOBAL CONFIGURATION AND SCHEMA ---

# Resolve project-rooted paths (works cross-platform)
BASE_DIR = Path(__file__).resolve().parent.parent
ENV = dotenv_values(BASE_DIR / ".env")

def get_env(key: str, default=None):
    val = ENV.get(key)
    return val if val not in (None, "") else default

ML_LIBRARY_ROOT = BASE_DIR / "ML_BOT" / "ML_Library"
TRAINED_MODELS_DIR = BASE_DIR / "Trained_Models"
SAMPLE_DATASET_PATH = BASE_DIR / "ML_BOT" / "Sample_Datasets" / "Mall_Customers.csv"

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

# --- 3. LLM INTERACTION (Using Gemini) ---

def get_model_suggestion_from_llama3(stats: str, filename: str, target_col: str, context: str) -> dict | None: 
    """
    Calls Gemini via google-generativeai, constraining the 'suggested_model' field to CamelCase names.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ google-generativeai library not found. Please run 'pip install google-generativeai'.")
        return None

    api_key = get_env("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY is not set in .env.")
        return None
    genai.configure(api_key=api_key)
    
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
    
    print("\n--- 🤖 Sending Analysis to Gemini... ---")
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        json_str = response.text
        print(f"✅ Gemini response received.")
        return json.loads(json_str)
    except json.JSONDecodeError:
        snippet = json_str[:200] if 'json_str' in locals() else ''
        print(f"❌ JSON Decode Error: Gemini did not return valid JSON. Start of response: {snippet}")
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

def execute_model_script(script_path: str,
                         data_filepath: str,
                         target_col: str | None,
                         model_output_dir: str,
                         auto_exit: bool = False,
                         timeout: int = 300):
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

    # 3. Execute the command
    try:
        # If auto_exit=True, we capture output and send "exit" to skip the prediction loop.
        if auto_exit:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                input="exit\n",
                timeout=timeout
            )
            print("\n✅ Model script exited successfully.")
            print(f"Model saved to: {pkl_full_path}")
            return {
                "pkl_path": pkl_full_path,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        else:
            # Interactive mode: inherit stdio so user can type into the prediction loop.
            result = subprocess.run(
                command,
                check=True,
                text=True
            )
            return {
                "pkl_path": pkl_full_path,
                "stdout": "",
                "stderr": "",
                "returncode": result.returncode,
            }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during script execution (Return Code {e.returncode})")
        return {
            "pkl_path": pkl_full_path,
            "stdout": e.stdout if auto_exit else "",
            "stderr": e.stderr if auto_exit else "",
            "returncode": e.returncode,
        }
    except subprocess.TimeoutExpired as e:
        print("❌ Error: Script execution timed out.")
        return {
            "pkl_path": pkl_full_path,
            "stdout": e.stdout if auto_exit else "",
            "stderr": e.stderr if auto_exit else "",
            "returncode": -1,
        }
    except FileNotFoundError:
        print("❌ Error: Python executable or script not found. Check paths.")
        return {
            "pkl_path": pkl_full_path,
            "stdout": "",
            "stderr": "Python executable or script not found.",
            "returncode": -1,
        }
        
# --- 6. MAIN ORCHESTRATOR FUNCTION ---

def orchestrate_ml_pipeline(dataset_path: str,
                            target_column: str | None,
                            user_context: str,
                            train: bool = False,
                            auto_exit: bool = True) -> dict:
    """
    Non-interactive orchestrator used by both CLI and UI.
    Returns a dictionary with suggestion details and (optionally) training output.
    """
    results: dict[str, str | dict | None] = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "context": user_context
    }

    df = read_dataset(dataset_path)
    if df is None:
        raise ValueError("Dataset could not be loaded.")

    file_name = Path(dataset_path).name
    stats_string = generate_stats(df)
    results["stats"] = stats_string

    llm_result = get_model_suggestion_from_llama3(stats_string, file_name, target_column or "", user_context)
    if llm_result is None:
        raise RuntimeError("LLM suggestion failed.")

    problem_type = llm_result.get('problem_type')
    suggested_model = llm_result.get('suggested_model')
    results["suggestion"] = llm_result

    model_script_name = f"{suggested_model}.py"
    
    script_path = find_model_script(
        library_path=str(ML_LIBRARY_ROOT),
        problem_type=problem_type,
        script_filename=model_script_name
    )
    results["script_path"] = script_path

    if train and script_path:
        train_result = execute_model_script(
            script_path=script_path,
            data_filepath=dataset_path,
            target_col=target_column,
            model_output_dir=str(TRAINED_MODELS_DIR),
            auto_exit=auto_exit
        )
        results["train_result"] = train_result

    return results


def _run_cli():
    """
    Simple CLI entrypoint preserving the previous behavior while allowing overrides.
    """
    print("="*50)
    print("✨ ML Model Orchestration Pipeline (CLI) ✨")
    print("="*50)

    dataset_path = input(f"Dataset Filepath (default: {SAMPLE_DATASET_PATH}): ").strip() or str(SAMPLE_DATASET_PATH)
    target_column = input("Enter the TARGET COLUMN name (or leave blank if unsupervised): ").strip() or None
    user_context = input("Enter your CONTEXT/GOAL for this dataset (e.g., 'Predict survival'): ").strip()
    if not user_context:
        user_context = "Analyze the data and suggest a fitting ML task."

    try:
        results = orchestrate_ml_pipeline(dataset_path, target_column, user_context, train=True, auto_exit=True)
        print(json.dumps(results.get("suggestion", {}), indent=2))
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


# --- 7. EXECUTION BLOCK (Final, Corrected) ---

if __name__ == "__main__":
    # Check 1: google-generativeai
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        print("FATAL ERROR: The 'google-generativeai' library is not installed.")
        print("Please install it using: pip install google-generativeai")
        sys.exit(1)
        
    # Check 2: tabulate (Corrected check for 'tabulate' dependency)
    try:
        pd.DataFrame({'A': [1]}).describe().to_markdown() 
    except ImportError:
        print("FATAL ERROR: The 'tabulate' library is missing (required by pandas.to_markdown()).")
        print("Please install it using: pip install tabulate")
        sys.exit(1)

    _run_cli()
