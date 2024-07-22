```python
import os
import subprocess
import random
import time
from typing import Dict, List, Tuple
from datetime import datetime
import logging
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, huggingface_hub
from huggingface_hub import InferenceClient, cached_download, Repository, HfApi
from IPython.display import display, HTML
import streamlit.components.v1 as components
import tempfile
import shutil

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
project_path = DEFAULT_PROJECT_PATH  # Default project path

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
            model_kwargs={"load_in_8bit": True},
        )
        # Fetch and store the model description
        api = HfApi()
        model_info = api.model_info(model_name)
        model_descriptions[model_name] = model_info.pipeline_tag
        return f"Successfully loaded model: {model_name}"
    except Exception as e:
        return f"Error loading model: {str(e)}"


def run_command(command: str, project_path: str = None) -> str:
    """Executes a shell command and returns the output."""
    try:
        if project_path:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        output, error = process.communicate()
        if error:
            return f"""Error: {error.decode('utf-8')}"""
        return output.decode("utf-8")
    except Exception as e:
        return f"""Error executing command: {str(e)}"""


def create_project(project_name: str, project_path: str = DEFAULT_PROJECT_PATH):
    """Creates a new Hugging Face project."""
    global repo, project_path
    try:
        if os.path.exists(project_path):
            return f"""Error: Directory '{project_path}' already exists!"""
        # Create the repository
        repo = Repository(local_dir=project_path, clone_from=None)
        repo.git_init()
        # Add basic files (optional, can customize this)
        with open(os.path.join(project_path, "README.md"), "w") as f:
            f.write(f"{project_name}\n\nA new Hugging Face project.")
        # Stage all changes
        repo.git_add(pattern="*")
        repo.git_commit(commit_message="Initial commit")
        project_path = os.path.join(project_path, project_name)  # Update project path
        return f"""Hugging Face project '{project_name}' created successfully at '{project_path}'"""
    except Exception as e:
        return f"""Error creating Hugging Face project: {str(e)}"""


def list_files(project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Lists files in the project directory."""
    try:
        files = os.listdir(project_path)
        if not files:
            return "Project directory is empty."
        return "\n".join(files)
    except Exception as e:
        return f"""Error listing project files: {str(e)}"""


def read_file(file_path: str, project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Reads and returns the content of a file in the project."""
    try:
        full_path = os.path.join(project_path, file_path)
        with open(full_path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"""Error reading file: {str(e)}"""


def write_file(file_path: str, content: str, project_path: str = DEFAULT_PROJECT_PATH):
    """Writes content to a file in the project."""
    try:
        full_path = os.path.join(project_path, file_path)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to '{full_path}'"
    except Exception as e:
        return f"""Error writing to file: {str(e)}"""


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
        return f"""Error previewing project: {str(e)}"""


def generate_response(
    message: str,
    history: List[Tuple[str, str]],
    agent_name: str,
    sys_prompt: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    """Generates a response using the loaded model."""
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


def run_chat(
    purpose: str,
    message: str,
    agent_name: str,
    sys_prompt: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    repetition_penalty: float,
    history: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Handles the chat interaction."""
    if not current_model:
        return [(history, history), "Please load a model first."]
    response = generate_response(
        message,
        history,
        agent_name,
        sys_prompt,
        temperature,
        max_new_tokens,
        top_p,
        repetition_penalty,
    )
    history.append((message, response))
    return [(history, history), response]


def update_model_dropdown(category):
    """Populates the model dropdown based on the selected category."""
    models = []
    api = HfApi()
    for model in api.list_models():
        if model.pipeline_tag == category:
            models.append(model.modelId)
    return gr.Dropdown.update(choices=models)


def display_model_description(model_name):
    """Displays the description of the selected model."""
    global model_descriptions
    if model_name in model_descriptions:
        return model_descriptions[model_name]
    else:
        return "Model description not available."


def load_selected_model(model_name):
    """Loads the selected model."""
    global current_model
    load_output = load_model(model_name)
    if current_model:
        return f"""Model '{model_name}' loaded successfully!"""
    else:
        return f"""Error loading model '{model_name}'"""


def create_project_handler(project_name):
    """Handles the creation of a new project."""
    return create_project(project_name)


def list_files_handler():
    """Handles the listing of files in the project directory."""
    return list_files(project_path)


def read_file_handler(file_path):
    """Handles the reading of a file in the project."""
    return read_file(file_path, project_path)


def write_file_handler(file_path, file_content):
    """Handles the writing of content to a file in the project."""
    return write_file(file_path, file_content, project_path)


def run_command_handler(command):
    """Handles the execution of a shell command."""
    return run_command(command, project_path)


def preview_handler():
    """Handles the preview of the project."""
    return preview(project_path)


def main():
    """Main function to launch the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("## IDEvIII: Your Hugging Face No-Code App Builder")
        # --- Model Selection ---
        with gr.Tab("Model"):
            model_categories = gr.Dropdown(
                choices=[
                    "Text Generation",
                    "Text Summarization",
                    "Code Generation",
                    "Translation",
                    "Question Answering",
                ],
                label="Model Category",
                value="Text Generation",
            )
            model_name = gr.Dropdown(
                choices=[],  # Initially empty, will be populated based on category
                label="Hugging Face Model Name",
            )
            load_button = gr.Button("Load Model")
            load_output = gr.Textbox(label="Output")
            model_description = gr.Markdown(label="Model Description")

            model_categories.change(
                fn=update_model_dropdown, inputs=model_categories, outputs=model_name
            )
            model_name.change(
                fn=display_model_description, inputs=model_name, outputs=model_description
            )
            load_button.click(
                load_selected_model, inputs=model_name, outputs=load_output
            )

        # --- Chat Interface ---
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                show_label=False,
                show_share_button=False,
                show_copy_button=True,
                likeable=True,
            )
            message = gr.Textbox(
                label="Enter your message", placeholder="Ask me anything!"
            )
            purpose = gr.Textbox(
                label="Purpose", placeholder="What is the purpose of this interaction?"
            )
            agent_name = gr.Textbox(
                label="Agent Name", value="Generic Agent", interactive=True
            )
            sys_prompt = gr.Textbox(
                label="System Prompt", max_lines=1, interactive=True
            )
            temperature = gr.Slider(
                label="Temperature",
                value=TEMPERATURE,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                interactive=True,
                info="Higher values produce more creative text.",
            )
            max_new_tokens = gr.Slider(
                label="Max new tokens",
                value=MAX_TOKENS,
                minimum=0,
                maximum=1048 * 10,
                step=64,
                interactive=True,
                info="The maximum number of new tokens to generate.",
            )
            top_p = gr.Slider(
                label="Top-p (nucleus sampling)",
                value=TOP_P,
                minimum=0,
                maximum=1,
                step=0.05,
                interactive=True,
                info="Higher values sample more low-probability tokens.",
            )
            repetition_penalty = gr.Slider(
                label="Repetition penalty",
                value=REPETITION_PENALTY,
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                interactive=True,
                info="Penalize repeated tokens.",
            )
            submit_button = gr.Button(value="Send")
            history = gr.State([])
            submit_button.click(
                run_chat,
                inputs=[
                    purpose,
                    message,
                    agent_name,
                    sys_prompt,
                    temperature,
                    max_new_tokens,
                    top_p,
                    repetition_penalty,
                    history,
                ],
                outputs=[chatbot, history],
            )

        # --- Project Management ---
        with gr.Tab("Project"):
            project_name = gr.Textbox(label="Project Name")
            create_project_button = gr.Button("Create Project")
            create_project_output = gr.Textbox(label="Output")
            list_files_button = gr.Button("List Files")
            list_files_output = gr.Textbox(label="Output")
            file_path = gr.Textbox(label="File Path")
            read_file_button = gr.Button("Read File")
            read_file_output = gr.Textbox(label="Output")
            file_content = gr.Textbox(label="File Content")
            write_file_button = gr.Button("Write File")
            write_file_output = gr.Textbox(label="Output")
            run_command_input = gr.Textbox(label="Command")
            run_command_button = gr.Button("Run Command")
            run_command_output = gr.Textbox(label="Output")
            preview_button = gr.Button("Preview")
            preview_output = gr.Textbox(label="Output")

            create_project_button.click(
                create_project_handler, inputs=project_name, outputs=create_project_output
            )
            list_files_button.click(
                list_files_handler, outputs=list_files_output
            )
            read_file_button.click(
                read_file_handler, inputs=file_path, outputs=read_file_output
            )
            write_file_button.click(
                write_file_handler,
                inputs=[file_path, file_content],
                outputs=write_file_output,
            )
            run_command_button.click(
                run_command_handler, inputs=run_command_input, outputs=run_command_output
            )
            preview_button.click(
                preview_handler, outputs=preview_output
            )

        # --- Custom Server Settings ---
        server_name = "0.0.0.0"  # Listen on available network interfaces
        server_port = 7606  # Choose an available port
        share_gradio_link = True  # Share a public URL for the app

        # --- Launch the Interface ---
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share_gradio_link,
        )


if __name__ == "__main__":
    main()