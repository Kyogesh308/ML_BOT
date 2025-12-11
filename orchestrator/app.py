import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from feature import (
    SAMPLE_DATASET_PATH,
    orchestrate_ml_pipeline,
)


st.set_page_config(page_title="ML Bot UI", layout="wide")

VALID_EMAIL = "Yotzimlbot@gmail.com"
VALID_PASSWORD = "yotziyotziyo"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
        :root {
            --mlb-bg: #0b1220;
            --mlb-panel: #111a2e;
            --mlb-card: #0f192b;
            --mlb-border: #1f2a41;
            --mlb-accent: #57d2ff;
            --mlb-accent-2: #83ffb1;
            --mlb-text: #e9edf5;
            --mlb-subtle: #b6bed3;
        }
        .stApp, .stMarkdown, .stText, .stTextInput, body {
            font-family: 'Space Grotesk', 'Inter', system-ui, -apple-system, sans-serif;
            color: var(--mlb-text);
        }
        .stApp {
            background: radial-gradient(circle at 10% 20%, #1a2b4b 0, rgba(17,26,46,0) 25%),
                        radial-gradient(circle at 80% 0%, #13365a 0, rgba(17,26,46,0) 25%),
                        var(--mlb-bg);
        }
        .mlb-hero {
            border: 1px solid var(--mlb-border);
            background: linear-gradient(135deg, rgba(87,210,255,0.15), rgba(17,26,46,0.7));
            border-radius: 18px;
            padding: 22px 24px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.28);
        }
        .mlb-hero-title {
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        .mlb-hero-sub {
            color: var(--mlb-subtle);
            font-size: 15px;
            margin-top: 4px;
        }
        .mlb-tag {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(87,210,255,0.12);
            color: var(--mlb-accent);
            border: 1px solid rgba(87,210,255,0.35);
            font-weight: 600;
            font-size: 12px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .mlb-card {
            border: 1px solid var(--mlb-border);
            background: var(--mlb-card);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.2);
        }
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, var(--mlb-accent), var(--mlb-accent-2));
            color: #0b1220;
            font-weight: 700;
            border: none;
            border-radius: 12px;
            padding: 12px 16px;
        }
        .stButton>button:hover { filter: brightness(1.05); }
        .stCheckbox>label, .stTextInput>label, .stTextArea>label {
            color: var(--mlb-subtle);
            font-weight: 600;
        }
        .stTextInput>div>input, .stTextArea>div>textarea {
            border-radius: 10px;
            border: 1px solid var(--mlb-border);
            background: #0b1325;
            color: var(--mlb-text);
        }
        .stFileUploader>label {
            color: var(--mlb-subtle);
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _require_login() -> bool:
    """Simple gate before loading the main UI."""
    _inject_styles()

    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        """
        <div class="mlb-hero">
            <div class="mlb-tag">Access</div>
            <div class="mlb-hero-title">ML Bot Portal</div>
            <div class="mlb-hero-sub">Enter the provided credentials to continue.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", placeholder="Email")
        password = st.text_input("Password", placeholder="Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if email == VALID_EMAIL and password == VALID_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Login successful. Loading the app...")
            # Re-run to load the main UI after authentication.
            rerun = getattr(st, "rerun", None)
            if callable(rerun):
                rerun()
            else:
                st.stop()
        else:
            st.error("Invalid email or password.")

    return False


if not _require_login():
    st.stop()

_inject_styles()

st.markdown(
    """
    <div class="mlb-hero">
        <div class="mlb-tag">ML Bot</div>
        <div class="mlb-hero-title">Dataset → Model Suggestion → (Optional) Train</div>
        <div class="mlb-hero-sub">Upload, add context, get an informed model suggestion, and train locally.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Data setup")
with st.container():
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("#### Dataset")
        uploaded_file = st.file_uploader(
            "Upload a dataset",
            type=["csv", "xlsx", "xls", "json", "parquet", "tsv", "txt"],
            help="File is stored temporarily on disk for processing.",
        )
        use_sample = st.checkbox("Use bundled sample dataset", value=not bool(uploaded_file))
    with right:
        st.markdown("#### Context")
        target_column = st.text_input(
            "Target column (leave empty for unsupervised)", placeholder="e.g., Outcome"
        )
        user_context = st.text_area(
            "Context / goal", placeholder="e.g., Predict customer churn"
        )
        run_training = st.checkbox("Train model after suggestion", value=False)

st.divider()
run_btn = st.button("Run pipeline", type="primary", use_container_width=True)


def _persist_upload(file) -> Path:
    uploads_dir = Path("tmp/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / file.name
    dest.write_bytes(file.getbuffer())
    return dest


def _load_model_bundle(pkl_path: str) -> dict:
    """Load a trained model bundle and normalize structure."""
    bundle = joblib.load(pkl_path)
    if not isinstance(bundle, (tuple, list)) or len(bundle) < 3:
        raise ValueError("Model bundle is not in the expected format.")
    model = bundle[0]
    scaler = bundle[1] if len(bundle) > 1 else None
    feature_columns = bundle[2] if len(bundle) > 2 else []
    feature_encoders = bundle[3] if len(bundle) > 3 else {}
    target_encoder = bundle[4] if len(bundle) > 4 else None
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns or [],
        "feature_encoders": feature_encoders or {},
        "target_encoder": target_encoder,
    }


def _predict_single(bundle: dict, values: dict[str, str]) -> str:
    """Run a single prediction given raw string inputs."""
    feature_columns = bundle.get("feature_columns", [])
    encoders = bundle.get("feature_encoders", {})
    scaler = bundle.get("scaler")
    model = bundle.get("model")
    target_encoder = bundle.get("target_encoder")

    ordered_vals = []
    for col in feature_columns:
        raw = values.get(col, "")
        if col in encoders:
            le = encoders[col]
            try:
                ordered_vals.append(le.transform([raw])[0])
            except Exception as exc:
                raise ValueError(f"Value '{raw}' is not valid for '{col}'.") from exc
        else:
            try:
                ordered_vals.append(float(raw))
            except ValueError as exc:
                raise ValueError(f"Column '{col}' expects a number.") from exc

    df = pd.DataFrame([ordered_vals], columns=feature_columns)
    X = scaler.transform(df) if scaler is not None else df
    pred = model.predict(X)[0]
    if target_encoder is not None:
        try:
            pred = target_encoder.inverse_transform([pred])[0]
        except Exception:
            pass
    return str(pred)


def _render_results(result: dict) -> None:
    suggestion = result.get("suggestion", {}) or {}
    st.markdown("### Results")

    st.markdown('<div class="mlb-card">', unsafe_allow_html=True)
    st.markdown("#### Suggested model")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Problem Type:** {suggestion.get('problem_type') or 'Not detected'}")
        st.markdown(f"**Suggested Model:** {suggestion.get('suggested_model') or 'N/A'}")
    with col_b:
        st.markdown(f"**Unnecessary Column:** {suggestion.get('unnecesarry_column_in_the_dataset') or 'None flagged'}")
    st.markdown("**Reasoning**")
    st.markdown(suggestion.get("reasoning") or "No reasoning provided.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="mlb-card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("#### Script location")
    st.write(result.get("script_path") or "Not found in ML library.")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Dataset stats (from pandas describe/info)"):
        st.text(result.get("stats", "No stats available."))

    train_result = result.get("train_result")
    if train_result:
        st.markdown('<div class="mlb-card" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown("#### Training output")
        st.write(f"Model file: {train_result.get('pkl_path')}")
        pkl_path = train_result.get("pkl_path")
        if pkl_path and Path(pkl_path).exists():
            with open(pkl_path, "rb") as f:
                st.download_button(
                    label="Download trained model (.pkl)",
                    data=f.read(),
                    file_name=Path(pkl_path).name,
                    mime="application/octet-stream",
                )
        else:
            st.info("Model file not found on disk; re-run training to regenerate.")
        with st.expander("stdout"):
            st.text(train_result.get("stdout") or "")
        with st.expander("stderr"):
            st.text(train_result.get("stderr") or "")
        st.markdown("</div>", unsafe_allow_html=True)

        # Inline prediction UI using the trained model bundle.
        if pkl_path and Path(pkl_path).exists():
            st.markdown('<div class="mlb-card" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown("#### Try a prediction")
            bundle = None
            bundle_key = f"bundle::{pkl_path}"
            if st.session_state.get(bundle_key):
                bundle = st.session_state[bundle_key]
            else:
                try:
                    bundle = _load_model_bundle(pkl_path)
                    st.session_state[bundle_key] = bundle
                except Exception as exc:
                    st.error(f"Could not load model for prediction: {exc}")

            if bundle:
                feature_columns = bundle.get("feature_columns", [])
                encoders = bundle.get("feature_encoders", {})

                if not feature_columns:
                    st.info("Feature metadata not available in model bundle; prediction UI disabled.")
                else:
                    with st.form("predict_form"):
                        user_inputs: dict[str, str] = {}
                        for col in feature_columns:
                            if col in encoders:
                                choices = list(getattr(encoders[col], "classes_", []))
                                user_inputs[col] = st.selectbox(
                                    col,
                                    options=choices or ["-"],
                                    help="Select a known category.",
                                )
                            else:
                                user_inputs[col] = st.text_input(col, placeholder="Numeric value")
                        submit_pred = st.form_submit_button("Predict")

                    if submit_pred:
                        try:
                            prediction = _predict_single(bundle, user_inputs)
                            st.success(f"Predicted value: {prediction}")
                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")

            st.markdown("</div>", unsafe_allow_html=True)


if run_btn:
    dataset_path: Path | None = None

    auto_exit = True  # Always auto-exit CLI prediction loops; UI handles prediction
    if uploaded_file:
        dataset_path = _persist_upload(uploaded_file)
    elif use_sample:
        dataset_path = SAMPLE_DATASET_PATH
    else:
        st.error("Please upload a dataset or select the sample dataset.")

    if dataset_path:
        with st.spinner("Running orchestration..."):
            try:
                result = orchestrate_ml_pipeline(
                    dataset_path=str(dataset_path),
                    target_column=target_column or None,
                    user_context=user_context or "Analyze the data and suggest a fitting ML task.",
                    train=run_training,
                    auto_exit=auto_exit,
                )
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
            else:
                st.session_state["last_result"] = result
                st.success("Pipeline completed.")

                _render_results(result)

# Re-render last successful run so prediction UI remains after rerun
if not run_btn and st.session_state.get("last_result"):
    _render_results(st.session_state["last_result"])
