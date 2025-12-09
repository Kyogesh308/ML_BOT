import argparse
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values
import pandas as pd

# --- GLOBAL CONFIGURATION AND SCHEMA ---

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV = dotenv_values(REPO_ROOT / ".env")


def get_env(key: str, default=None):
    return ENV.get(key) if ENV.get(key) not in (None, "") else default


ML_LIBRARY_ROOT = Path(
    get_env("ML_LIBRARY_ROOT", REPO_ROOT / "ML_BOT" / "ML_Library")
).resolve()
TRAINED_MODELS_DIR = Path(
    get_env("TRAINED_MODELS_DIR", REPO_ROOT / "Trained_Models")
).resolve()
DEFAULT_DATASET_PATH = Path(
    get_env(
        "DEFAULT_DATASET_PATH",
        REPO_ROOT / "ML_BOT" / "Sample_Datasets" / "Mall_Customers.csv",
    )
).resolve()

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
        if file_extension == ".csv":
            df = pd.read_csv(filepath)
        elif file_extension in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        elif file_extension == ".json":
            df = pd.read_json(filepath)
        elif file_extension == ".txt" or file_extension == ".tsv":
            # Assuming space or tab-separated for txt/tsv
            df = pd.read_csv(filepath, sep=r"\s+|,", engine="python")
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
    try:
        stats_output.write(df.describe(include="all").to_markdown())
    except ValueError:
        # Fallback for edge cases like empty frames
        stats_output.write("No describe() output available.")
    except ImportError:
        stats_output.write(df.describe(include="all").to_string())
    stats_output.write("\n\n--- Missing Value Counts ---\n")
    stats_output.write(df.isnull().sum().to_markdown(numalign="left", stralign="left"))
    return stats_output.getvalue()

# --- 3. LLM INTERACTION (Using Gemini) ---

def get_model_suggestion_from_llama3(
    stats: str, filename: str, target_col: str, context: str
) -> dict | None:
    """
    Calls Gemini (hosted) and expects JSON back for problem/model suggestion.
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
    classification_models = ", ".join(MODEL_SCHEMA["Classification"])
    regression_models = ", ".join(MODEL_SCHEMA["Regression"])
    clustering_models = ", ".join(MODEL_SCHEMA["Clustering"])
    dr_models = ", ".join(MODEL_SCHEMA["DimensionalityReduction"])
    
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
    
    print("\n--- 🤖 Sending Analysis to Gemini... ---")
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        raw_text = response.text
        print("✅ Gemini response received.")
        return json.loads(raw_text)
    except json.JSONDecodeError:
        snippet = raw_text[:200] if "raw_text" in locals() else ""
        print(f"❌ JSON Decode Error: Gemini did not return valid JSON. Raw start: {snippet}...")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during Gemini call: {e}")
        return None

# --- 4. MODEL SEARCH FUNCTION ---

def find_model_script(library_path, problem_type, script_filename: str) -> str | None: 
    """
    Finds the model script in the ML_Library using problem_type and the model filename.
    """
    problem_dir_path = os.path.join(str(library_path), problem_type)

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

def execute_model_script(script_path: str, data_filepath: str, target_col: str | None, model_output_dir: str) -> dict:
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
        return {"status": "ok", "model_path": pkl_full_path}

    except subprocess.CalledProcessError as e:
        # This catches errors if the child script crashes during training or prediction.
        print(f"❌ Error during script execution (Return Code {e.returncode})")
        print("Please check the terminal output for the script's internal error messages.")
        return {"status": "error", "message": f"Return code {e.returncode}"}
    except FileNotFoundError:
        print("❌ Error: Python executable or script not found. Check paths.")
        return {"status": "error", "message": "Python executable or script not found"}
        
# --- 6. MAIN ORCHESTRATOR FUNCTION ---

def orchestrate_pipeline_entry(
    dataset_path: str,
    target_column: str | None,
    user_context: str,
    run_training: bool,
    emit_json: bool = False,
) -> dict | None:
    """Core orchestration logic that can be called from CLI or imported."""
    dataset_path = Path(dataset_path).expanduser().resolve()
    result: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "target_column": target_column,
        "context": user_context,
        "run_training": run_training,
        "stats": None,
        "llm_result": None,
        "script_path": None,
        "training": None,
        "status": "pending",
    }

    print("=" * 50)
    print("✨ ML Model Orchestration Pipeline ✨")
    print(f"Dataset Filepath: {dataset_path}")
    print("=" * 50)

    print("\n--- Starting Data Analysis ---")
    df = read_dataset(dataset_path)
    if df is None:
        result["status"] = "error"
        result["error"] = "Dataset could not be read."
    else:
        file_name = dataset_path.name
        stats_string = generate_stats(df)
        result["stats"] = stats_string
        print("The Calculated stats are: ", stats_string)

        llm_result = get_model_suggestion_from_llama3(
            stats_string, file_name, target_column, user_context
        )
        result["llm_result"] = llm_result

        if llm_result is None:
            result["status"] = "llm_failed"
            result["error"] = "LLM call failed or returned invalid JSON."
        else:
            problem_type = llm_result.get("problem_type")
            suggested_model = llm_result.get("suggested_model")

            if not problem_type or not suggested_model:
                result["status"] = "llm_missing_fields"
                result["error"] = "LLM response did not contain problem_type or suggested_model."
            else:
                print("\n" + "=" * 50)
                print(f"🤖 LLM Problem Assessment")
                print("=" * 50)
                print(f"  Problem Type: **{problem_type}**")
                print(f"  Suggested Model: **{suggested_model}**")
                print(f"  Reasoning: {llm_result.get('reasoning')}")
                print(f"  Unnecessary Columns: {llm_result.get('unnecesarry_column_in_the_dataset')}")
                print("=" * 50 + "\n")

                model_script_name = f"{suggested_model}.py"

                script_path = find_model_script(
                    library_path=ML_LIBRARY_ROOT,
                    problem_type=problem_type,
                    script_filename=model_script_name,
                )
                result["script_path"] = script_path

                if script_path is None:
                    result["status"] = "model_not_found"
                    result["error"] = f"Model script {model_script_name} not found for {problem_type}."
                else:
                    if run_training:
                        training_result = execute_model_script(
                            script_path=script_path,
                            data_filepath=str(dataset_path),
                            target_col=target_column,
                            model_output_dir=TRAINED_MODELS_DIR,
                        )
                        result["training"] = training_result
                        if training_result.get("status") != "ok":
                            result["status"] = "training_failed"

                    if result["status"] == "pending":
                        result["status"] = "ok"

    if emit_json:
        print(json.dumps(result, default=str))
    else:
        print("\n✨ ML Orchestration Pipeline Complete. Check the Trained_Models directory! ✨")

    return result


def orchestrate_ml_pipeline():
    """Legacy wrapper kept for backwards compatibility."""
    return orchestrate_pipeline_entry(
        dataset_path=str(DEFAULT_DATASET_PATH),
        target_column=None,
        user_context="Analyze the data and suggest a fitting ML task.",
        run_training=False,
        emit_json=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ML pipeline suggestion/training for a dataset."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--target",
        "-t",
        default=None,
        help="Target column for supervised tasks. Leave empty for unsupervised.",
    )
    parser.add_argument(
        "--context",
        "-c",
        default="Analyze the data and suggest a fitting ML task.",
        help="Context/goal for the dataset.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Also run the suggested training script (may be interactive).",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print the result object as JSON (for API/automation).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    orchestrate_pipeline_entry(
        dataset_path=args.dataset,
        target_column=args.target,
        user_context=args.context,
        run_training=args.train,
        emit_json=args.json_output,
    )
