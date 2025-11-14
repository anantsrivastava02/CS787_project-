import csv
import subprocess
import sys
import os
import re

# Set UTF-8 encoding for console output
os.environ['PYTHONIOENCODING'] = 'utf-8'

OLLAMA_PATH = r"C:\Users\anant\AppData\Local\Programs\Ollama\ollama.exe"

def download_model(model_name):
    """Download the specified Ollama model."""
    print(f"Downloading model: {model_name}...")
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "pull", model_name],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='replace'
        )
        print(f"Model downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"Error downloading model: {error_msg.encode('utf-8', errors='replace').decode('utf-8')}")
        return False
    except FileNotFoundError:
        print("Error: Ollama is not installed or not in PATH")
        return False

def generate_response(model_name, instruction_prompt, user_prompt):
    """Generate a response from the model."""
    full_prompt = f"{instruction_prompt}\n\n{user_prompt}"

    try:
        result = subprocess.run(
            [OLLAMA_PATH, "run", model_name, full_prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,  # 5 minute timeout per prompt
            encoding='utf-8',
            errors='replace'
        )
        response = result.stdout.strip()

        # For deepseek-r1, strip out the thinking process
        if 'deepseek-r1' in model_name.lower():
            # Remove everything between "Thinking..." and "...done thinking."
            response = re.sub(r'Thinking\.\.\..*?\.\.\.done thinking\.\s*', '', response, flags=re.DOTALL)
            response = response.strip()

        return response
    except subprocess.TimeoutExpired:
        return "ERROR: Request timed out"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr}"
    except Exception as e:
        return f"ERROR: {str(e)}"

def process_prompts(input_csv, model_name, instruction_prompt):
    """Process all prompts from CSV and save results."""
    # Sanitize filename by replacing invalid characters
    safe_name = model_name.replace('/', '_').replace(':', '_')
    output_csv = f"{safe_name}_results.csv"

    # Read input CSV
    prompts = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get('prompt', '').strip()
            # Skip empty rows
            if prompt:
                prompts.append(prompt)

    print(f"Found {len(prompts)} non-empty prompts to process")

    # Process each prompt and save results
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing prompt {i}/{len(prompts)}...")
        print(f"Prompt preview: {prompt[:100]}...")

        response = generate_response(model_name, instruction_prompt, prompt)
        results.append({
            'prompt': prompt,
            'model_response': response
        })

        print(f"Response generated ({len(response)} characters)")

    # Write results to output CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'model_response'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved to: {output_csv}")
    return output_csv

if __name__ == "__main__":
    MODELS = ["deepseek-r1:latest"]
    INPUT_CSV = "prompts.csv"
    INSTRUCTION_PROMPT = """You are a story generator, given a context or a start of an event make a story.
Your story must be complete so write as much as you can in 400 to 500 words."""

    # Step 1: Download all models
    print("=== Downloading models ===")
    for model in MODELS:
        if not download_model(model):
            print(f"Warning: Failed to download {model}, but continuing...")
        print()

    # Step 2: Process prompts with each model
    print("\n=== Processing prompts ===")
    for model in MODELS:
        print(f"\n--- Processing with {model} ---")
        output_file = process_prompts(INPUT_CSV, model, INSTRUCTION_PROMPT)
        print(f"✓ Completed {model}: {output_file}")

    print(f"\n✓ All done! Processed {len(MODELS)} models.")
