import os
import shutil

# --- CONFIGURATION ---
# Default paths (can be overridden when calling the function)
DEFAULT_MODEL_PATH = "Trained_Models/model.pkl" 
DEFAULT_APP_TEMPLATE = "app.py"
DEFAULT_LAUNCHER_TEMPLATE = "app_launcher.py"
DEFAULT_APP_TEMPLATE_FOR_REGRESSION = "rapp.py"
DEFAULT_LAUNCHER_TEMPLATE_FOR_REGRESSION = "rapp_launcher.py"
def create_user_package(output_folder_name, model_path=DEFAULT_MODEL_PATH, problem_type = str):
    """
    Call this function from your main pipeline to generate the package instantly.
    
    Args:
        output_folder_name (str): The name of the folder to create (e.g. "Download_For_User_X")
        model_path (str): Path to the specific .pkl file to bundle.
    """
    print(f"--- Packaging: {output_folder_name} ---")

    # 1. Clean/Create Directory
    if os.path.exists(output_folder_name):
        shutil.rmtree(output_folder_name)
    os.makedirs(output_folder_name)

    # 2. Copy the Model
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find model at {model_path}")
        return
    
    # Destination is always 'model.pkl' so app.py can find it easily
    shutil.copy(model_path, os.path.join(output_folder_name, os.path.basename(model_path)))
    print("✅ Model copied.")

    # 3. Copy the Python Templates (app.py & app_launcher.py)
    # These must exist in your root folder
    if 'regression' not in problem_type.lower(): 
        for file in [DEFAULT_APP_TEMPLATE, DEFAULT_LAUNCHER_TEMPLATE]:
            if os.path.exists(file):
                shutil.copy(file, os.path.join(output_folder_name, file))
                print(f"✅ {file} copied.")
            else:
                print(f"❌ Warning: {file} template not found in root directory.")
    else:
        for file in [DEFAULT_APP_TEMPLATE_FOR_REGRESSION, DEFAULT_LAUNCHER_TEMPLATE_FOR_REGRESSION]:
            if os.path.exists(file):
                shutil.copy(file, os.path.join(output_folder_name, file))
                print(f"✅ {file} copied.")
            else:
                print(f"❌ Warning: {file} template not found in root directory.")

    # 4. Create Requirements File
    reqs = "streamlit\npandas\nnumpy\nscikit-learn\njoblib"
    with open(os.path.join(output_folder_name, "requirements.txt"), "w") as f:
        f.write(reqs)
    
    readme = """Hi user, double click the Run_App.bat file. your terminal will be opened, nothing
    to be scared of, it is just trying to run the streamlit app on your local host, as streamlit is 
    a web based platform, no need to panic, your app will be opened, and you can make predictions,
    see insights and thanks for using ML_BOT by Yotzeron. """
    with open(os.path.join(output_folder_name, "README"), "w") as f:
        f.write(readme)

    # 5. Create 'Run_App.bat' (The instant launcher)
    if "regression" not in problem_type.lower():
        bat_code = """@echo off
echo Starting App...
streamlit run app.py
pause
"""
    else: 
        bat_code = """@echo off
echo Starting App...
streamlit run rapp.py
pause
"""
    with open(os.path.join(output_folder_name, "Run_App.bat"), "w") as f:
        f.write(bat_code)
    print("✅ Run_App.bat created.")

    print(f"\n✨ Package Ready at: {os.path.abspath(output_folder_name)}")
    print("-------------------------------------------------------")

