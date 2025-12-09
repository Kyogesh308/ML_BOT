# import sys
# import subprocess
# def execute_script(filename: str, filepath, target, pkl_file) -> None:
#     """Executes a Python script using the current interpreter."""
#     print(f"\nExecuting script: {filename}")
#     try:
#         result = subprocess.run(
#             [sys.executable, filename, filepath, target, pkl_file]
#         )
#         print("--- Script Output (stdout) ---")
#         print(result.stdout.strip())
#         print("------------------------------")

#     except subprocess.TimeoutExpired:
#         print(f"\n⏱Script '{filename}' timed out after 300 seconds.")
#     except subprocess.CalledProcessError as e:
#         print(f"\n Script Execution Error in {filename}:")
#         print(f"Return Code: {e.returncode}")
#         print("--- Stderr ---")
#         print(e.stderr.strip())
#         print("--------------")

#     except Exception as e:
#         print(f"\nAn unexpected error occurred during execution: {e}")

# execute_script("ML_BOT\ML_Library\Classification\Trees_Ensembles\RandomForestClassifier.py", "ML_BOT\Sample_Datasets\diabetes.csv"
#                ,"Outcome", "Trained.pkl")


import os
from pathlib import Path

def find_model_script(library_path, problem_type, script_filename="model_script.py"):
    problem_dir_path = os.path.join(library_path, problem_type)

    if not os.path.isdir(problem_dir_path):
        print(f"Error: Directory '{problem_dir_path}' not found.")
        return None

    for root, _, files in os.walk(problem_dir_path):
        if script_filename in files:
            model_script_path = os.path.join(root, script_filename)
            print(f"✅ Found script for {problem_type} at: {model_script_path}")
            return model_script_path
    
    print(f"❌ Could not find '{script_filename}' in any subdirectory of '{problem_type}'.")
    return None


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    library = repo_root / "ML_BOT" / "ML_Library"
    find_model_script(str(library), "Classification", script_filename="LogisticRegression.py")
