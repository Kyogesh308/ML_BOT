import streamlit.web.cli as stcli
import os, sys
import multiprocessing

def resolve_path(path):
    if getattr(sys, 'frozen', False):
        basedir = os.path.dirname(sys.executable)
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

if __name__ == '__main__':
    # 1. STOP Infinite Loop
    multiprocessing.freeze_support()
    
    # 2. Configure Environment
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    
    # 3. Find app.py
    app_path = resolve_path("rapp.py")
    
    # 4. Run Streamlit directly
    sys.argv = ["streamlit", "run", app_path, "--global.developmentMode=false"]
    sys.exit(stcli.main())