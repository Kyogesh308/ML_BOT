# classification app
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION ---
st.set_page_config(page_title="Prediction System", layout="centered")

def find_model_file():
    """
    Returns the relative path of the only .pkl file in the current directory.
    Assumes exactly one .pkl file exists.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(current_dir):
        if f.endswith('.pkl'):
            return os.path.join(current_dir, f)
    return None  # fallback if no .pkl file exists



def load_pipeline(model_path):
    """
    Loads the specific 5-component tuple from the given path.
    """
    if not os.path.exists(model_path):
        st.error(f"❌ File not found: '{model_path}'")
        return None

    try:
        data = joblib.load(model_path)
        if not isinstance(data, (tuple, list)) or len(data) < 5:
            st.error("❌ PKL file format mismatch. Expected a tuple of 5 components.")
            return None
        return data 
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {e}")
        return None

def run_app(model_path):
    """
    The main utility function that renders the UI for a specific model file.
    Args:
        model_path (str): Relative or absolute path to the .pkl file.
    """
    st.title("🤖 Advanced Prediction App")
    st.caption(f"Running model: `{os.path.basename(model_path)}`")

    pipeline = load_pipeline(model_path)
    if pipeline is None:
        return

    # Unpack components
    model, scaler, feature_columns, feature_encoders, target_encoder = pipeline

    # --- DYNAMIC FORM GENERATION ---
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

        submitted = st.form_submit_button("Run Analysis", type="primary")

    # --- PREDICTION LOGIC ---
    if submitted:
        try:
            # 1. Transform Categorical Inputs (Text -> Integer)
            processed_data = []
            for col_name in feature_columns:
                raw_val = user_inputs[col_name]
                
                if col_name in feature_encoders:
                    encoder = feature_encoders[col_name]
                    encoded_val = encoder.transform([raw_val])[0]
                    processed_data.append(encoded_val)
                else:
                    processed_data.append(raw_val)

            # 2. Create DataFrame & Scale
            X_df = pd.DataFrame([processed_data], columns=feature_columns)
            X_scaled = scaler.transform(X_df)

            # 3. Predict & Decode
            prediction_idx = model.predict(X_scaled)[0]
            prediction_label = target_encoder.inverse_transform([prediction_idx])[0]

            # 4. Display Results
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Predicted Class", str(prediction_label))
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_scaled)
                confidence = np.max(probs) * 100
                c2.metric("Confidence", f"{confidence:.2f}%")
                
        except Exception as e:
            st.error(f"Pipeline Error: {e}")
    
    # --- 5. DIAGNOSTICS ---
    with st.expander("🛠️ Diagnostics"):
        st.write(f"**Loaded File:** {model_path}")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Expected Columns:** {feature_columns}")

if __name__ == "__main__":
    # 1. Try to find a model file automatically
    detected_path = find_model_file()
    
    if detected_path:
        run_app(detected_path)
    else:
        st.error("❌ No .pkl file found in this directory. Please place your model file here.")