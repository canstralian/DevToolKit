
import os
import subprocess
import random
import time
from typing import Dict, List, Tuple
from datetime import datetime
import logging
import huggingface_hub as hfApi
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient, cached_download, Repository
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
        return output.decode('utf-8')
    except Exception as e:
        return f"Error executing command: {str(e)}"

def create_project(project_name: str, project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Creates a new Hugging Face project."""
    global repo
    try:
        if os.path.exists(project_path):
            return f"Error: Directory '{project_path}' already exists!"
        # Create the repository
        repo = Repository(local_dir=project_path, clone_from=None)
        repo.git_init()
        # Add basic files (optional, can customize this)
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

def read_file(filepath: str, project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Reads and returns the content of a file in the project."""
    try:
        full_path = os.path.join(project_path, filepath)
        with open(full_path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(filepath: str, content: str, project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Writes content to a file in the project."""
    try:
        full_path = os.path.join(project_path, filepath)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to '{full_path}'"
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
        with gr.Tab("Model Selection"):
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
            agent_name = gr.Textbox(label="Agent Name", value="Generic Agent", interactive=True)
            sys_prompt = gr.Textbox(label="System Prompt", max_lines=1, interactive=True)
            temperature = gr.Slider(label="Temperature", value=TEMPERATURE, minimum=0.0, maximum=1.0, step=0.05, interactive=True, info="Higher values produce more random results")
            max_new_tokens = gr.Slider(label="Max new tokens", value=MAX_TOKENS, minimum=0, maximum=1048 * 10, step=64, interactive=True, info="The maximum number of new tokens")
            top_p = gr.Slider(label="Top-p (nucleus sampling)", value=TOP_P, minimum=0, maximum=1, step=0.05, interactive=True, info="Higher values sample more low-probability tokens")
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
                        repetition_penalty=repetition_penalty
                    )
                    return response.text.strip()
                response_text = generate_response(message, history, agent_name, sys_prompt, temperature, max_new_tokens, top_p, repetition_penalty)
                history.append((message, response_text))
                return history, history

            submit_button.click(run_chat, inputs=[purpose, message, agent_name, sys_prompt, temperature, max_new_tokens, top_p, repetition_penalty, history], outputs=[chatbot, history])

        # --- Project Management ---
        with gr.Tab("Project Management"):
            project_name_input = gr.Textbox(label="Project Name", placeholder="Enter project name")
            create_project_button = gr.Button("Create Project")
            project_output = gr.Textbox(label="Output")

            def create_project_action(project_name):
                return create_project(project_name)

            create_project_button.click(create_project_action, inputs=project_name_input, outputs=project_output)

            list_files_button = gr.Button("List Files")
            list_files_output = gr.Textbox(label="Files")

            def list_files_action():
                return list_files()

            list_files_button.click(list_files_action, outputs=list_files_output)

            file_path_input = gr.Textbox(label="File Path", placeholder="Enter file path")
            read_file_button = gr.Button("Read File")
            read_file_output = gr.Textbox(label="File Content")

            def read_file_action(file_path):
                return read_file(file_path)

            read_file_button.click(read_file_action, inputs=file_path_input, outputs=read_file_output)

            write_file_button = gr.Button("Write File")
            file_content_input = gr.Textbox(label="File Content", placeholder="Enter file content")

            def write_file_action(file_path, file_content):
                return write_file(file_path, file_content)

            write_file_button.click(write_file_action, inputs=[file_path_input, file_content_input], outputs=project_output)

            run_command_input = gr.Textbox(label="Command", placeholder="Enter command")
            run_command_button = gr.Button("Run Command")
            run_command_output = gr.Textbox(label="Command Output")

            def run_command_action(command):
                return run_command(command)

            run_command_button.click(run_command_action, inputs=run_command_input, outputs=run_command_output)

            preview_button = gr.Button("Preview Project")
            preview_output = gr.Textbox(label="Preview URL")

            def preview_action():
                return preview()

            preview_button.click(preview_action, outputs=preview_output)

        # Custom server settings
        server_name = "0.0.0.0"  # Listen on all available network interfaces
        server_port = 5000  # Choose an available port
        share_gradio_link = True  # Share a public URL for the app

        # Launch the interface
        demo.launch(server_name=server_name, server_port=server_port, share=share_gradio_link)

if __name__ == "__main__":
    main()