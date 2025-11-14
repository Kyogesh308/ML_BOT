import sys
import subprocess
import ollama
from subprocess import Popen, PIPE

OLLAMA_MODEL = "llama3"

def generate_script_content(model: str, prompt_template: str) -> str:
    """
    Calls Ollama to generate a Python script based on the provided model and prompt template.
    """
    print(f"1. Calling Ollama client with model '{model}' to generate script...")

    refined_prompt = f"ONLY output the Python code, with no explanations, markdown, or extra text. {prompt_template}"

    try:
        client = ollama.Client()
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=refined_prompt,
            options={'temperature': 0.0}
        )
        raw_text = response['response'].strip()
        print("   Ollama output received via Python client.")
        return raw_text
    except ollama.OllamaAPIError as e:
        print(f"\nError: Ollama API call failed. Is Ollama running and is the model '{model}' downloaded?")
        print(f"  Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during Ollama generation: {e}")
        sys.exit(1)

def extract_code_block(raw_text: str) -> str:
    """
    Strips markdown fences from Ollama's output and returns clean Python code.
    """
    
    code_content = raw_text.strip()
    original_content = code_content
    fences = ['```python', '```py', '```']

    for fence in fences:
        if code_content.lower().startswith(fence):
            code_content = code_content[len(fence):].lstrip()
            break

    if code_content.rstrip().endswith('```'):
        last_fence_index = code_content.rfind('```')
        if last_fence_index != -1:
            code_content = code_content[:last_fence_index].rstrip()

    code_content = code_content.strip()

    if code_content != original_content:
        print("2. Extracted and cleaned code by aggressively stripping markdown fences.")
    elif original_content.startswith('```') or original_content.endswith('```'):
        print("2. Warning: Markdown fences detected but stripping failed to change content.")
    else:
        print("2. Assuming raw output is the script content (no fences found/removed).")

    return code_content

def save_script(filename: str, content: str) -> None:
    """
    Saves the extracted code content to a Python file.
    """
    print(f"3. Saving code to file: {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

import subprocess
import sys

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