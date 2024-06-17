import os
import subprocess
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from llama_cpp import Llama, LlamaCppPythonProvider, LlamaCppAgent
from llama_cpp.llama_cpp_agent import get_messages_formatter_type, get_context_by_model
from io import StringIO
import tempfile

# --- Global Variables ---
CURRENT_PROJECT = {}  # Store project data (code, packages, etc.)
MODEL_OPTIONS = {
    "CodeQwen": "Qwen/CodeQwen1.5-7B-Chat-GGUF",
    "Codestral": "bartowski/Codestral-22B-v0.1-GGUF",
    "AutoCoder": "bartowski/AutoCoder-GGUF",
}
MODEL_FILENAMES = {
    "CodeQwen": "codeqwen-1_5-7b-chat-q6_k.gguf",
    "Codestral": "Codestral-22B-v0.1-Q6_K.gguf",
    "AutoCoder": "AutoCoder-Q6_K.gguf",
}
HUGGING_FACE_REPO_URL = "https://huggingface.co/spaces/acecalisto3/DevToolKit"
PROJECT_ROOT = "projects"
AGENT_DIRECTORY = "agents"

# Global state to manage communication between Tool Box and Workspace Chat App
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'terminal_history' not in st.session_state:
    st.session_state.terminal_history = []
if 'workspace_projects' not in st.session_state:
    st.session_state.workspace_projects = {}
if 'available_agents' not in st.session_state:
    st.session_state.available_agents = []
if 'current_state' not in st.session_state:
    st.session_state.current_state = {
        'toolbox': {},
        'workspace_chat': {}
    }

# --- Load NLP Pipelines ---
classifier = pipeline("text-classification", model="facebook/bart-large-mnli")

# --- Load the model and tokenizer ---
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token=os.environ.get("huggingface_token"))
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token=os.environ.get("huggingface_token"))

# --- Utility Functions ---
def install_and_import(package_name):
    """Installs a package using pip and imports it."""
    subprocess.check_call(["pip", "install", package_name])
    return importlib.import_module(package_name)

def extract_package_name(input_str):
    """Extracts the package name from a PyPI URL or pip command."""
    if input_str.startswith("https://pypi.org/project/"):
        return input_str.split("/")[-2]
    elif input_str.startswith("pip install "):
        return input_str.split(" ")[2]
    else:
        return input_str

def create_interface_from_input(input_str):
    """Creates a Gradio interface with buttons for functions from a package."""
    try:
        package_name = extract_package_name(input_str)
        module = install_and_import(package_name)

        # Handle Flask application context if needed
        if 'flask' in sys.modules or 'flask_restful' in sys.modules:
            app = Flask(__name__)
            with app.app_context():
                functions = [getattr(module, name) for name in dir(module) if callable(getattr(module, name))]
        else:
            functions = [getattr(module, name) for name in dir(module) if callable(getattr(module, name))]

        function_list = [(func.__name__, func) for func in functions if not func.__name__.startswith("_")]
        return function_list, f"Interface for `{package_name}` created."

    except Exception as e:
        return [], str(e)

def execute_pip_command(command, add_message):
    """Executes a pip command and streams the output."""
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            add_message("System", f"


\n{output.strip()}\n

time.sleep(0.1)  # Simulate delay for more realistic streaming
    rc = process.poll()
    return rc

def generate_text(input_text):
    """Generates text using the loaded language model."""
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=500, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- AI Agent Functions ---
def analyze_user_intent(user_input):
    """Classifies the user's intent based on their input."""
    classification = classifier(user_input)
    return classification[0]['label']

def generate_mini_app_ideas(theme):
    """Generates mini-app ideas based on the user's theme."""
    if theme.lower() == "productivity":
        return [
            "Idea-to-Codebase Generator",
            "Automated GitHub Repo Manager",
            "AI-Powered IDE"
        ]
    elif theme.lower() == "creativity":
        return [
            "Brainstorming Assistant",
            "Mood Board Generator",
            "Writing Assistant"
        ]
    elif theme.lower() == "well-being":
        return [
            "Meditation Guide",
            "Mood Tracker",
            "Sleep Tracker"
        ]
    else:
        return ["No matching mini-apps found. Try a different theme."]

def generate_app_code(app_name, app_description, model_name, history):
    """Generates code for the selected mini-app using the specified GGUF model."""
    prompt = f"Write a Python script for a {app_description} named {app_name} using Gradio and Streamlit:"
    agent = get_agent(model_name)
    generated_code = agent.chat(prompt, history)
    return generated_code

def execute_terminal_command(command):
    """Executes a terminal command and returns the output."""
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        return result.strip(), None
    except subprocess.CalledProcessError as e:
        return e.output.strip(), str(e)

def install_package(package_name):
    """Installs a package using pip."""
    output, error = execute_terminal_command(f"pip install {package_name}")
    if error:
        return f"Error installing package: {error}"
    else:
        return f"Package `{package_name}` installed successfully."

def get_project_data():
    """Returns the current project data."""
    return CURRENT_PROJECT

def update_project_data(key, value):
    """Updates the project data."""
    CURRENT_PROJECT[key] = value

def handle_chat(input_text, history):
    """Handles user input in the chat interface."""
    def add_message(sender, message):
        history.append((sender, message))

    add_message("User", input_text)

    if input_text.startswith("pip install ") or input_text.startswith("https://pypi.org/project/"):
        package_name = extract_package_name(input_text)
        add_message("System", f"Installing `{package_name}`...")
        result = install_package(package_name)
        add_message("System", result)
        update_project_data("packages", CURRENT_PROJECT.get("packages", []) + [package_name])
        return history, dynamic_functions

    # --- AI Agent Interaction ---
    if USER_INTENT is None:
        add_message("System", analyze_user_intent(input_text))
        add_message("System", "What kind of mini-app do you have in mind?")
    elif not MINI_APPS:
        add_message("System", "Here are some ideas:")
        for idea in generate_mini_app_ideas(input_text):
            add_message("System", f"- {idea}")
        add_message("System", "Which one would you like to build?")
    elif CURRENT_APP["name"] is None:
        selected_app = input_text
        app_description = next((app for app in MINI_APPS if selected_app in app), None)
        if app_description:
            add_message("System", f"Generating code for {app_description}...")
            code = generate_app_code(selected_app, app_description, "CodeQwen", history)  # Use CodeQwen by default
            add_message("System", f"


python\n{code}\n

add_message("System", "Code generated! What else can I do for you?")
            update_project_data("code", code)
            update_project_data("app_name", selected_app)
            update_project_data("app_description", app_description)
        else:
            add_message("System", "Please choose from the provided mini-app ideas.")
    else:
        add_message("System", "You already have an app in progress. Do you want to start over?")

    return history, dynamic_functions

# --- Prebuilt Tools ---
def generate_code_tool(input_text, history):
    """Prebuilt tool for code generation."""
    code = generate_app_code("MyTool", "A tool to do something", "CodeQwen", history)  # Use CodeQwen by default
    return f"


python\n{code}\n

def analyze_code_tool(input_text, history):
    """Prebuilt tool for code analysis."""
    agent = get_agent("Codestral")
    analysis = agent.chat(input_text, history)
    return analysis

# --- Streamlit Interface ---
st.title("AI4ME: Your Personal AI App Workshop")
st.markdown("## Let's build your dream app together! 🤖")

# --- Hugging Face Token Input ---
huggingface_token = st.text_input("Enter your Hugging Face Token", type="password", key="huggingface_token")
os.environ["huggingface_token"] = huggingface_token

# --- Chat Interface ---
chat_history = []
chat_input = st.text_input("Tell me your idea...", key="chat_input")
if chat_input:
    chat_history, dynamic_functions = handle_chat(chat_input, chat_history)
    for sender, message in chat_history:
        st.markdown(f"**{sender}:** {message}")

# --- Code Execution and Deployment ---
if CURRENT_APP["code"]:
    st.markdown("## Your App Code:")
    code_area = st.text_area("Your App Code", value=CURRENT_APP["code"], key="code_area")

    st.markdown("## Deploy Your App (Coming Soon!)")
    # Add deployment functionality here using Streamlit's deployment features.
    # For example, you could use Streamlit's `st.button` to trigger deployment.

    # --- Code Execution ---
    st.markdown("## Run Your App:")
    if st.button("Execute Code"):
        try:
            # Use Hugging Face's text-generation pipeline for code execution
            inputs = tokenizer(code_area, return_tensors="pt")
            output = model.generate(**inputs, max_length=500, num_return_sequences=1)
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            st.success(f"Code executed successfully!\n{output}")
        except Exception as e:
            st.error(f"Error executing code: {e}")

    # --- Code Editing ---
    st.markdown("## Edit Your Code:")
    if st.button("Edit Code"):
        try:
            # Use Hugging Face's text-generation pipeline for code editing
            prompt = f"Improve the following Python code:\n


python\n{code_area}\n

inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(**inputs, max_length=500, num_return_sequences=1)
            edited_code = tokenizer.decode(output[0], skip_special_tokens=True).split("


python\n")[1].split("\n

st.success(f"Code edited successfully!\n{edited_code}")
            update_project_data("code", edited_code)
            code_area.value = edited_code
        except Exception as e:
            st.error(f"Error editing code: {e}")

# --- Prebuilt Tools ---
st.markdown("## Prebuilt Tools:")
with st.expander("Generate Code"):
    code_input = st.text_area("Enter your code request:", key="code_input")
    if st.button("Generate"):
        code_output = generate_code_tool(code_input, chat_history)
        st.markdown(code_output)

with st.expander("Analyze Code"):
    code_input = st.text_area("Enter your code:", key="analyze_code_input")
    if st.button("Analyze"):
        analysis_output = analyze_code_tool(code_input, chat_history)
        st.markdown(analysis_output)

# --- Additional Features ---
# Add features like:
# - Code editing
# - Integration with external APIs
# - Advanced AI agents for more complex tasks
# - User account management