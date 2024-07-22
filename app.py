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

# --- Constants ---
PREFIX = """Date: {date_time_str}
Purpose: {purpose}
Agent Name: {agent_name}
"""

LOG_PROMPT = """Prompt:
{content}
"""

LOG_RESPONSE = """Response:
{resp}
"""

# --- Functions ---
def format_prompt(message: str, history: List[Tuple[str, str]], max_history_turns: int = 2) -> str:
    prompt = ""
    for user_prompt, bot_response in history[-max_history_turns:]:
        prompt += f"Human: {user_prompt}\nAssistant: {bot_response}\n"
    prompt += f"Human: {message}\nAssistant:"
    return prompt

def generate_response(
    prompt: str,
    history: List[Tuple[str, str]],
    agent_name: str = "Generic Agent",
    sys_prompt: str = "",
    temperature: float = TEMPERATURE,
    max_new_tokens: int = MAX_TOKENS,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
) -> str:
    global current_model
    if current_model is None:
        return "Error: Please load a model first."

    date_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_prompt = PREFIX.format(
        date_time_str=date_time_str,
        purpose=sys_prompt,
        agent_name=agent_name
    ) + format_prompt(prompt, history)

    if VERBOSE:
        logging.info(LOG_PROMPT.format(content=full_prompt))

    response = current_model(
        full_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True
    )[0]['generated_text']

    assistant_response = response.split("Assistant:")[-1].strip()

    if VERBOSE:
        logging.info(LOG_RESPONSE.format(resp=assistant_response))

    return assistant_response

def load_hf_model(model_name: str):
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

def execute_command(command: str, project_path: str = None) -> str:
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

def create_hf_project(project_name: str, project_path: str = DEFAULT_PROJECT_PATH):
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

def list_project_files(project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Lists files in the project directory."""
    try:
        files = os.listdir(project_path)
        if not files:
            return "Project directory is empty."
        return "\n".join(files)
    except Exception as e:
        return f"Error listing project files: {str(e)}"

def read_file_content(file_path: str, project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Reads and returns the content of a file in the project."""
    try:
        full_path = os.path.join(project_path, file_path)
        with open(full_path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_to_file(file_path: str, content: str, project_path: str = DEFAULT_PROJECT_PATH) -> str:
    """Writes content to a file in the project."""
    try:
        full_path = os.path.join(project_path, file_path)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to '{file_path}'"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

def preview_project(project_path: str = DEFAULT_PROJECT_PATH):
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
        gr.Markdown("## FragMixt: Your Hugging Face No-Code App Builder")

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
                load_output = load_hf_model(model_name)
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
                response = generate_response(message, history, agent_name, sys_prompt, temperature, max_new_tokens, top_p, repetition_penalty)
                history.append((message, response))
                return history, history

            submit_button.click(run_chat, inputs=[purpose, message, agent_name, sys_prompt, temperature, max_new_tokens, top_p, repetition_penalty, history], outputs=[chatbot, history])

        # --- Project Management ---
        with gr.Tab("Project"):
            project_name = gr.Textbox(label="Project Name", placeholder="MyHuggingFaceApp")
            create_project_button = gr.Button("Create Hugging Face Project")
            project_output = gr.Textbox(label="Output", lines=5)
            file_content = gr.Code(label="File Content", language="python", lines=20)
            file_path = gr.Textbox(label="File Path (relative to project)", placeholder="src/main.py")
            read_button = gr.Button("Read File")
            write_button = gr.Button("Write to File")
            command_input = gr.Textbox(label="Terminal Command", placeholder="pip install -r requirements.txt")
            command_output = gr.Textbox(label="Command Output", lines=5)
            run_command_button = gr.Button("Run Command")
            preview_button = gr.Button("Preview Project")

            create_project_button.click(create_hf_project, inputs=[project_name], outputs=project_output)
            read_button.click(read_file_content, inputs=file_path, outputs=file_content)
            write_button.click(write_to_file, inputs=[file_path, file_content], outputs=project_output)
            run_command_button.click(execute_command, inputs=command_input, outputs=command_output)
            preview_button.click(preview_project, outputs=project_output)

    demo.launch()

if __name__ == "__main__":
    main()