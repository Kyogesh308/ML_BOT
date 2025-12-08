import os
from pathlib import Path

import streamlit as st

from Smallest_Prototype.feature import (
    SAMPLE_DATASET_PATH,
    orchestrate_ml_pipeline,
)


st.set_page_config(page_title="ML Bot UI", layout="wide")
st.title("ML Bot • Dataset → Model Suggestion → (Optional) Train")

st.markdown(
    "Upload a dataset, provide context and the target column (or leave blank for unsupervised), "
    "and run the pipeline. Training uses your local Ollama + sklearn scripts."
)

uploaded_file = st.file_uploader(
    "Upload dataset",
    type=["csv", "xlsx", "xls", "json", "parquet", "tsv", "txt"],
    help="File is stored temporarily on disk for processing.",
)

use_sample = st.checkbox("Use bundled sample dataset", value=not bool(uploaded_file))

target_column = st.text_input(
    "Target column (leave empty for unsupervised)", placeholder="e.g., Outcome"
)
user_context = st.text_area(
    "Context / goal", placeholder="e.g., Predict customer churn"
)

col1, col2 = st.columns(2)
with col1:
    run_training = st.checkbox("Train model after suggestion", value=False)
with col2:
    auto_exit = st.checkbox(
        "Auto-exit prediction loop (send 'exit' after training)", value=True
    )

run_btn = st.button("Run pipeline")


def _persist_upload(file) -> Path:
    uploads_dir = Path("tmp/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / file.name
    dest.write_bytes(file.getbuffer())
    return dest


if run_btn:
    dataset_path: Path | None = None

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
                st.success("Pipeline completed.")

                suggestion = result.get("suggestion", {}) or {}
                st.subheader("Suggested Model")
                st.write(f"**Problem Type:** {suggestion.get('problem_type')}")
                st.write(f"**Suggested Model:** {suggestion.get('suggested_model')}")
                st.write(f"**Reasoning:** {suggestion.get('reasoning')}")
                st.write(f"**Unnecessary Column:** {suggestion.get('unnecesarry_column_in_the_dataset')}")

                st.subheader("Script Location")
                st.write(result.get("script_path") or "Not found in ML library.")

                with st.expander("Dataset stats (from pandas describe/info)"):
                    st.text(result.get("stats", "No stats available."))

                train_result = result.get("train_result")
                if train_result:
                    st.subheader("Training Output")
                    st.write(f"Model file: {train_result.get('pkl_path')}")
                    with st.expander("stdout"):
                        st.text(train_result.get("stdout") or "")
                    with st.expander("stderr"):
                        st.text(train_result.get("stderr") or "")
