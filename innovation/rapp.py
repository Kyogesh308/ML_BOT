import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys
import sklearn

# --- CRITICAL IMPORTS FOR UNPICKLING ---
# Importing Regressors just in case joblib needs them for reconstruction
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="Regression Prediction System", layout="centered")

def find_model_file():
    """
    Auto-detects a .pkl file in the current directory.
    Returns the absolute path of the first one found.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = [f for f in os.listdir(current_dir) if f.endswith('.pkl')]
    
    if not files:
        return None, current_dir
    
    return os.path.join(current_dir, files[0]), current_dir

def load_pipeline(model_path):
    """
    Loads the specific 4-component tuple for REGRESSION.
    Structure: (model, scaler, feature_columns, feature_encoders)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File not found: '{model_path}'")

    data = joblib.load(model_path)
    
    # Verify structure
    if not isinstance(data, (tuple, list)):
        raise ValueError(f"Format Error: Expected a Tuple, found '{type(data).__name__}'.")
    
    # REGRESSION CHECK: Expecting 4 components
    if len(data) < 4:
        raise ValueError(f"Structure Error: Expected 4 components (Model, Scaler, Cols, Encoders), found {len(data)}.")
        
    return data 

def run_app():
    st.title("📈 Regression Prediction App")

    # --- 1. FIND MODEL ---
    model_path, current_dir = find_model_file()
    
    if not model_path:
        st.error("❌ No .pkl file found!")
        st.info("Please copy your regression model .pkl file into this folder.")
        return

    st.caption(f"Loaded Model: `{os.path.basename(model_path)}`")

    # --- 2. LOAD PIPELINE ---
    try:
        pipeline = load_pipeline(model_path)
        # Unpack components (REGRESSION: Only 4 items)
        model, scaler, feature_columns, feature_encoders = pipeline[0], pipeline[1], pipeline[2], pipeline[3]
    except Exception as e:
        st.error("❌ Failed to load model.")
        st.code(str(e), language="bash")
        return

    # --- 3. DYNAMIC FORM GENERATION ---
    user_inputs = {}
    
    with st.form("input_form"):
        st.write("### Enter Input Data")
        cols = st.columns(2)
        
        for i, col_name in enumerate(feature_columns):
            col_ui = cols[i % 2]
            
            # CHECK: Is this column categorical? (Does it have an encoder?)
            if col_name in feature_encoders:
                # It's categorical -> Show a Dropdown
                encoder = feature_encoders[col_name]
                options = list(encoder.classes_)
                user_inputs[col_name] = col_ui.selectbox(f"{col_name}", options=options)
            else:
                # It's numerical -> Show a Number Box
                user_inputs[col_name] = col_ui.number_input(f"{col_name}", value=0.0)

        submitted = st.form_submit_button("Run Prediction", type="primary")

    # --- 4. PREDICTION LOGIC ---
    if submitted:
        try:
            # A. Transform Categorical Inputs (Text -> Integer)
            processed_data = []
            for col_name in feature_columns:
                raw_val = user_inputs[col_name]
                
                if col_name in feature_encoders:
                    encoder = feature_encoders[col_name]
                    encoded_val = encoder.transform([raw_val])[0]
                    processed_data.append(encoded_val)
                else:
                    processed_data.append(raw_val)

            # B. Create DataFrame & Scale
            X_df = pd.DataFrame([processed_data], columns=feature_columns)
            X_scaled = scaler.transform(X_df)

            # C. Predict (Regression: Direct numeric output)
            prediction_val = model.predict(X_scaled)[0]

            # D. Display Results
            st.divider()
            st.metric("Predicted Value", f"{prediction_val:.4f}")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # --- 5. DIAGNOSTICS ---
    with st.expander("🛠️ Diagnostics"):
        st.write(f"**Loaded File:** {model_path}")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Expected Columns:** {feature_columns}")

if __name__ == "__main__":
    run_app()