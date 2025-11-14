import sys
import subprocess
def execute_script(filename: str, filepath, target, pkl_file) -> None:
    """Executes a Python script using the current interpreter."""
    print(f"\nExecuting script: {filename}")
    try:
        result = subprocess.run(
            [sys.executable, filename, filepath, target, pkl_file]
        )
        print("--- Script Output (stdout) ---")
        print(result.stdout.strip())
        print("------------------------------")

    except subprocess.TimeoutExpired:
        print(f"\n⏱Script '{filename}' timed out after 300 seconds.")
    except subprocess.CalledProcessError as e:
        print(f"\n Script Execution Error in {filename}:")
        print(f"Return Code: {e.returncode}")
        print("--- Stderr ---")
        print(e.stderr.strip())
        print("--------------")

    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")

execute_script("ML_BOT\ML_Library\Classification\Trees_Ensembles\RandomForestClassifier.py", "ML_BOT\Sample_Datasets\diabetes.csv"
               ,"Outcome", "Trained.pkl")