import os
import subprocess
import random
import time
from typing import Dict, List, Tuple
from datetime import datetime
import logging

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient, cached_download, Repository, HfApi
from IPython.display import display, HTML
import streamlit.components.v1 as components

# --- Configuration ---
VERBOSE = True
MAX_HISTORY = 5
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.8
REPETITION_PENALTY = 1.5
DEFAULT_PROJECT_PATH = "./my-hf-project"  # Default project directory

# --- Logging Setup ---
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Global Variables ---
current_model = None  # Store the currently loaded model
repo = None  # Store the Hugging Face Repository object
model_descriptions = {}  # Store model descriptions

# --- Functions ---
def load_model(model_name: str):
    """Loads a language model and fetches its description."""
    global current_model, model_descriptions
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        current_model = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            model_kwargs={"load_in_8bit": True}
        )

        # Fetch and store the model description
        api = HfApi()
        model_info = api.model_info(model_name)
        model_descriptions[model_name] = model_info.pipeline_tag
        return f"Successfully loaded model: {model_name}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def model_selection():
    st.title("Model Selection")
    st.write("Select a model to use for code generation:")
    models = ["distilbert", "t5", "codellama-7b", "geminai-1.5b"]
    selected_model = st.selectbox("Select a model:", models)
    if selected_model:
        model = load_model(selected_model)
        if model:
            st.write(f"Model {selected_model} imported successfully!")
            return model
        else:
            st.write(f"Error importing model {selected_model}.")
    return None

def run_command(command: str, project_path: str = None) -> str:
    """Executes a shell command and returns the output."""
    try:
        if project_path:
            process = subprocess.Popen(command, shell=True, cwd=project_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            return f"Error: {error.decode('utf-8')}"
        return output.decode("utf-8")
    except Exception as e:
        return f"Error executing command: {str(e)}"

def create_project(project_name: str, project_path: str = DEFAULT_PROJECT_PATH):
    """Creates a new Hugging Face project."""
    global repo
    try:
        if os.path.exists(project_path):
            return f"Error: Directory '{project_path}' already exists!"
        # Create the repository
        repo = Repository(local_dir=project_path, clone_from=None)
        repo.git_init()

        # Add basic files (optional, you can customize this)
        with open(os.path.join(project_path, "README.md"), "w") as f:
            f.write(f"# {project_name}\n\nA new Hugging Face project.")

        # Stage all changes
        repo.git_add(pattern="*")
        repo.git_commit(commit_message="Initial commit")

        return f"Hugging Face project '{project_name}' created successfully at '{project_path}'"
    except Exception as e:
        return f"Error creating Hugging Face project: {str(e)}"

def list_files(project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Lists files in the project directory."""
    try:
        files = os.listdir(project_path)
        if not files:
            return "Project directory is empty."
        return "\n".join(files)
    except Exception as e:
        return f"Error listing project files: {str(e)}"

def read_file(file_path: str, project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Reads and returns the content of a file in the project."""
    try:
        full_path = os.path.join(project_path, file_path)
        with open(full_path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path: str, content: str, project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Writes content to a file in the project."""
    try:
        full_path = os.path.join(project_path, file_path)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to '{file_path}'"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

def preview(project_path: str = DEFAULT_PROJECT_PATH):
    """Provides a preview of the project, if applicable."""
    # Assuming a simple HTML preview for now
    try:
        index_html_path = os.path.join(project_path, "index.html")
        if os.path.exists(index_html_path):
            with open(index_html_path, "r") as f:
                html_content = f.read()
            display(HTML(html_content))
            return "Previewing 'index.html'"
        else:
            return "No 'index.html' found for preview."
    except Exception as e:
        return f"Error previewing project: {str(e)}"

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## IDEvIII: Your Hugging Face No-Code App Builder")

        # --- Model Selection ---
        with gr.Tab("Model"):
            # --- Model Dropdown with Categories ---
            model_categories = gr.Dropdown(
                choices=["Text Generation", "Text Summarization", "Code Generation", "Translation", "Question Answering"],
                label="Model Category",
                value="Text Generation"
            )
            model_name = gr.Dropdown(
                choices=[],  # Initially empty, will be populated based on category
                label="Hugging Face Model Name",
            )
            load_button = gr.Button("Load Model")
            load_output = gr.Textbox(label="Output")
            model_description = gr.Markdown(label="Model Description")

            # --- Function to populate model names based on category ---
            def update_model_dropdown(category):
                models = []
                api = HfApi()
                for model in api.list_models():
                    if model.pipeline_tag == category:
                        models.append(model.modelId)
                return gr.Dropdown.update(choices=models)

            # --- Event handler for category dropdown ---
            model_categories.change(
                fn=update_model_dropdown,
                inputs=model_categories,
                outputs=model_name,
            )

            # --- Event handler to display model description ---
            def display_model_description(model_name):
                global model_descriptions
                if model_name in model_descriptions:
                    return model_descriptions[model_name]
                else:
                    return "Model description not available."

            model_name.change(
                fn=display_model_description,
                inputs=model_name,
                outputs=model_description,
            )

            # --- Event handler to load the selected model ---
            def load_selected_model(model_name):
                global current_model
                load_output = load_model(model_name)
                if current_model:
                    return f"Model '{model_name}' loaded successfully!"
                else:
                    return f"Error loading model '{model_name}'"

            load_button.click(load_selected_model, inputs=model_name, outputs=load_output)

        # --- Chat Interface ---
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True)
            message = gr.Textbox(label="Enter your message", placeholder="Ask me anything!")
            purpose = gr.Textbox(label="Purpose", placeholder="What is the purpose of this interaction?")
            agent_name = gr.Dropdown(label="Agents", choices=["Generic Agent"], value="Generic Agent", interactive=True)
            sys_prompt = gr.Textbox(label="System Prompt", max_lines=1, interactive=True)
            temperature = gr.Slider(label="Temperature", value=TEMPERATURE, minimum=0.0, maximum=1.0, step=0.05, interactive=True, info="Higher values produce more diverse outputs")
            max_new_tokens = gr.Slider(label="Max new tokens", value=MAX_TOKENS, minimum=0, maximum=1048 * 10, step=64, interactive=True, info="The maximum numbers of new tokens")
            top_p = gr.Slider(label="Top-p (nucleus sampling)", value=TOP_P, minimum=0.0, maximum=1, step=0.05, interactive=True, info="Higher values sample more low-probability tokens")
            repetition_penalty = gr.Slider(label="Repetition penalty", value=REPETITION_PENALTY, minimum=1.0, maximum=2.0, step=0.05, interactive=True, info="Penalize repeated tokens")
            submit_button = gr.Button(value="Send")
            history = gr.State([])

            def run_chat(purpose: str, message: str, agent_name: str, sys_prompt: str, temperature: float, max_new_tokens: int, top_p: float, repetition_penalty: float, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
                if not current_model:
                    return [(history, history), "Please load a model first."]
            
            def generate_response(message, history, agent_name, sys_prompt, temperature, max_new_tokens, top_p, repetition_penalty):
    if not current_model:
        return "Please load a model first."

    conversation = [{"role": "system", "content": sys_prompt}]
    for message, response in history:
        conversation.append({"role": "user", "content": message})
        conversation.append({"role": "assistant", "content": response})
    conversation.append({"role": "user", "content": message})

    response = current_model.generate(
        conversation,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    return response.text.strip()

def create_project(project_name):
    try:
        repo_name = get_full_repo_name(project_name, token=HfApi().token)
        repo = HfFolder.create_repo(repo_name, exist_ok=True)
        repo.save_data("README.md", f"# {project_name}")
        return f"Created project '{project_name}' on Hugging Face Hub."
    except Exception as e:
        return f"Error creating project: {str(e)}"

def read_file(file_path):
    if not os.path.exists(file_path):
        return f"File '{file_path}' does not exist."

    try:
        with open(file_path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"

def write_file(file_path, file_content):
    try:
        with open(file_path, "w") as file:
            file.write(file_content)
        return f"Wrote to file '{file_path}' successfully."
    except Exception as e:
        return f"Error writing to file '{file_path}': {str(e)}"

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Command '{command}' failed with exit code {result.returncode}:\n{result.stderr}"
    except Exception as e:
        return f"Error running command '{command}': {str(e)}"


def preview():
    # Get the current working directory
    cwd = os.getcwd()

    # Create a temporary directory for the preview
    temp_dir = tempfile.mkdtemp()

    try:
        # Copy the project files to the temporary directory
        shutil.copytree(cwd, temp_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        # Change to the temporary directory
        os.chdir(temp_dir)

        # Find the main Python file (e.g., app.py, main.py)
        main_file = next((f for f in os.listdir(".") if f.endswith(".py")), None)

        if main_file:
            # Run the main Python file to generate the preview
            subprocess.run(["streamlit", "run", main_file], check=True)

            # Get the preview URL
            preview_url = components.get_url(main_file)

            # Change back to the original working directory
            os.chdir(cwd)

            # Return the preview URL
            return preview_url
        else:
            return "No main Python file found in the project."
    except Exception as e:
        return f"Error generating preview: {str(e)}"
    finally:
        # Remove the temporary directory
        shutil.rmtree(temp_dir)

    # Customize the launch settings
    server_name = "0.0.0.0"  # Listen on all available network interfaces
    server_port = 7860  # Choose an available port
    share_gradio_link = True  # Share a public URL for the app

    # Launch the interface
    demo.launch(server_name=server_name, server_port=server_port, share=share_gradio_link)

if __name__ == "__main__":
    main()