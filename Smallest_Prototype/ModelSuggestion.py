import json
import ollama
import os
import io
import pandas as pd
import sys

def model_suggestion(filename, dataset: pd.DataFrame, target, context):
    
    print("the name of the file: ", filename)
    
    print("\nDataset has been loaded!")
    
    # --- Dynamic Target Analysis ---
    problem_type = "unknown"
    if target and target in dataset.columns:
        target_series = dataset[target].dropna()
        if target_series.dtype in ['float64', 'float32', 'int64', 'int32'] and len(target_series.unique()) > 20:
            problem_type = "regression"
        elif len(target_series.unique()) == 2:
            problem_type = "binary_classification"
        elif 2 < len(target_series.unique()) < 50:
            problem_type = "multiclass_classification"
        elif target is None:
            problem_type = "clustering" # Assuming unsupervised if no target specified

    print("\n Extracting Information and Description of dataset.")

    description = dataset.describe().to_string()
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    information = buffer.getvalue()
    rows, columns = dataset.shape
    column_names = dataset.columns.tolist()
    print("\nCentral Tendencies, rows, columns, calculated.")
    print("\nAll the dataset stats are calculated")

    print(f"\nProblem Type Detected (Heuristic): {problem_type}")
    print("Moving towards ollama to suggest a model.")

    # --- Enforce JSON structure and determinism ---
    prompt = f"""The AutoML Prompt
    You are an expert AutoML system. Your sole function is to analyze the provided dataset statistics and recommend the optimal machine learning model.
    The primary problem type is likely: "{problem_type}".

    Based on the statistics, first verify the problem_type (regression, binary_classification, multiclass_classification, or clustering).
    Then, suggest the single most robust scikit-learn model for this problem.
    Finally, analyze the feature statistics (describe() output) to identify any column that has zero effect (zero variance, constant value, or missing correlation signal).

    You must provide your response only in the following JSON format. Do not include any explanatory text, greetings, or apologies before or after the JSON object.

    Dataset Statistics:
    Number of Rows: {rows}
    Number of Columns: {columns}
    All Column Names: {column_names}
    DataFrame info() Output (Data Types & Non-Nulls):
    {information}
    DataFrame describe() Output (Statistical Summary):
    {description}
    Context from the user: {context}
    
    Target Datatype: {dataset[target].dtype if target in dataset.columns else 'None'}
    
    Required JSON Output Format (Strictly provide only JSON output without any prefix, suffic tokens) ONLY return a valid JSON object. Do NOT include markdown, commentary, explanations, or formatting. Do NOT wrap the JSON in triple backticks. Just return the JSON object:

    {{
    "problem_type": "{problem_type}",
    "suggested_model": "string",
    "reasoning": "string",
    "unncesarry_column_in_the_dataset": "string"
    }}"""

    response = " "
    try:
        response = ollama.generate(
            model='llama3', 
            prompt=prompt,
            format="json",
            options={'temperature': 0.0} # FIX 1: Enforce Determinism
        )
        
        llm_output = response.get("response", "").strip()
        print("Ollama's raw response received.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    model = ""
    problem_type_llm = ""
    print("\nOllama's response: ")

    try:
        # FIX 2: Handle malformed JSON and missing keys gracefully
        data = json.loads(llm_output)    
        problem_type_llm = data.get('problem_type', 'UNKNOWN_TYPE')
        model = data.get('suggested_model', 'ErrorModel')
        print(json.dumps(data, indent=2))
        print(f"\nSuggested Model: {model}")
        print(f"Problem Type: {problem_type_llm}")
        print("Now we will move towards training a model!!! ")
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}. Raw LLM output was:\n{llm_output}")
        sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error during output processing: {e}")
        sys.exit(1)

    return model, problem_type_llm