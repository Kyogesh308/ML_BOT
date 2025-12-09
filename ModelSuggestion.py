import json
import os
import io
from pathlib import Path

import google.generativeai as genai
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent
ENV = dotenv_values(REPO_ROOT / ".env")

def model_suggestion(filename, dataset, target, context):
    
    print("the name of the file: ", filename)
    
    print("\nDataset has been loaded!")
    features = dataset.iloc[:, :-1]

    print("\n Extracting Information and Description of dataset.")

    description = dataset.describe().to_string()

    buffer = io.StringIO()
    dataset.info(buf=buffer)
    information = buffer.getvalue()
    rows, columns = dataset.shape
    column_names = dataset.columns.tolist()
    print("\nCentral Tendencies, rows, columns, calculated.")
    print("\nAll the dataset stats are calculated")

    print("Moving towards Gemini to suggest a model.")
    print("Gemini's response: ")
    datatype = None
    if target!=None:
        datatype = dataset[target].dtype
    

    #creating a prompt for the Gemini model to suggest a model to train the model
    prompt = f"""The AutoML Prompt
    You are an expert AutoML system. Your sole function is to analyze the provided dataset statistics and recommend the optimal machine learning model.

    Based on the statistics, first determine the problem_type by analyzing the Target Column.

    If the target column is continuous (e.g., float), the problem is "regression".

    If the target column has two unique discrete values (e.g., int, bool, or object/string), the problem is "binary_classification".

    If the target column has more than two but a limited number of unique discrete values, the problem is "multiclass_classification".

    Then, suggest a robust machine learning model for this problem.
    
    Also check from the {column_names}, if there is any column that has zero effect on the target column, if any. 

    You must provide your response only in the following JSON format. Do not include any explanatory text, greetings, or apologies before or after the JSON object.

    Dataset Statistics:

    Number of Rows: {rows}

    Number of Columns: {columns}

    All Column Names:

    {column_names}
    DataFrame info() Output (Data Types & Non-Nulls):

    {information}
    DataFrame describe() Output (Statistical Summary):

    {description}
    
    Context from the user: 
    {context}
    
    Target Datatype:
    {datatype}
    
    Required JSON Output Format (Strictly provide only JSON output without any prefix, suffic tokens) ONLY return a valid JSON object. Do NOT include markdown, commentary, explanations, or formatting. Do NOT wrap the JSON in triple backticks. Do NOT say anything before or after the JSON. Just return the JSON object:

    JSON

    {{
    "problem_type": "string",
    "suggested_model": "string",
    "reasoning": "string",
    "unncesarry_column_in_the_dataset": "string"
    }}"""

    response = " "
    api_key = ENV.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set.")
        return ""
    genai.configure(api_key=api_key)

    try:
        model_gen = genai.GenerativeModel("gemini-1.5-flash")
        resp_obj = model_gen.generate_content(prompt)
        llm_output = resp_obj.text or ""
        print(llm_output, "\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

    model = ""

    try:
        data = json.loads(llm_output)    
        problem_type = data['problem_type']
        model = data['suggested_model']
        print(f"Suggested Model: {model}")
        print(f"Problem Type: {problem_type}")
        print("Now we will move towards training a model!!! ")
    except Exception as e:
        print("JSON parsing failed:", e)    
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON response from Gemini.")
    except KeyError:
        print("Error: The JSON response is missing one of the expected keys.")
    return model

    
