import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, RagRetriever, AutoModelForSeq2SeqLM
import os
import subprocess
import black
from pylint import lint
from io import StringIO
import sys
import torch
from huggingface_hub import hf_hub_url, cached_download, HfApi

# Access Hugging Face API key from secrets
hf_token = st.secrets["hf_token"]
if not hf_token:
    st.error("Hugging Face API key not found. Please make sure it is set in the secrets.")

HUGGING_FACE_REPO_URL = "https://huggingface.co/spaces/acecalisto3/DevToolKit"
PROJECT_ROOT = "projects"
AGENT_DIRECTORY = "agents"
AVAILABLE_CODE_GENERATIVE_MODELS = ["bigcode/starcoder", "Salesforce/codegen-350M-mono", "microsoft/CodeGPT-small"]

# Global state to manage communication between Tool Box and Workspace Chat App
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'terminal_history' not in st.session_state:
    st.session_state.terminal_history = []
if 'workspace_projects' not in st.session_state:
    st.session_state.workspace_projects = {}
if 'available_agents' not in st.session_state:
    st.session_state.available_agents = []

# Initialize the session state variable as an empty string
st.session_state.user_input = ""

# Create the text input widget
user_input = st.text_input("Enter your text:", "Initial text")

# Use the get() method to get the current value of the widget and update it
if st.button("Update"):
    st.session_state.user_input = user_input
    
# AI Guide Toggle
ai_guide_level = st.sidebar.radio("AI Guide Level", ["Full Assistance", "Partial Assistance", "No Assistance"])

class AIAgent:
    def __init__(self, name, description, skills):
        self.name = name
        self.description = description
        self.skills = skills
        self._hf_api = HfApi()  # Initialize HfApi here

    def create_agent_prompt(self):
        skills_str = '\n'.join([f"* {skill}" for skill in self.skills])
        agent_prompt = f"""
As an elite expert developer, my name is {self.name}. I possess a comprehensive understanding of the following areas:
{skills_str}

I am confident that I can leverage my expertise to assist you in developing and deploying cutting-edge web applications. Please feel free to ask any questions or present any challenges you may encounter.
"""
        return agent_prompt

    def autonomous_build(self, chat_history, workspace_projects, project_name, selected_model, hf_token):
        summary = "Chat History:\n" + "\n".join([f"User: {u}\nAgent: {a}" for u, a in chat_history])
        summary += "\n\nWorkspace Projects:\n" + "\n".join([f"{p}: {details}" for p, details in workspace_projects.items()])
        next_step = "Based on the current state, the next logical step is to implement the main application logic."
        return summary, next_step

    def deploy_built_space_to_hf(self):
        # Assuming you have a function that generates the space content
        space_content = generate_space_content(project_name)
        repository = self._hf_api.create_repo(
            repo_id=project_name, 
            private=True,
            token=hf_token,
            exist_ok=True,
            space_sdk="streamlit"
        )
        self._hf_api.upload_file(
            path_or_fileobj=space_content,
            path_in_repo="app.py",
            repo_id=project_name,
            repo_type="space",
            token=hf_token
        )
        return repository

    def has_valid_hf_token(self):
        return self._hf_api.whoami(token=hf_token) is not None

def process_input(input_text):
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium", tokenizer="microsoft/DialoGPT-medium")
    response = chatbot(input_text, max_length=50, num_return_sequences=1)[0]['generated_text']
    return response

def run_code(code):
    try:
        result = subprocess.run(code, shell=True, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

def workspace_interface(project_name):
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
        st.session_state.workspace_projects[project_name] = {'files': []}
        return f"Project '{project_name}' created successfully."
    else:
        return f"Project '{project_name}' already exists."

def add_code_to_workspace(project_name, code, file_name):
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(project_path):
        return f"Project '{project_name}' does not exist."
    
    file_path = os.path.join(project_path, file_name)
    with open(file_path, "w") as file:
        file.write(code)
    st.session_state.workspace_projects[project_name]['files'].append(file_name)
    return f"Code added to '{file_name}' in project '{project_name}'."

def display_chat_history(chat_history):
    return "\n".join([f"User: {u}\nAgent: {a}" for u, a in chat_history])

def display_workspace_projects(workspace_projects):
    return "\n".join([f"{p}: {details}" for p, details in workspace_projects.items()])

def generate_space_content(project_name):
    # Logic to generate the Streamlit app content based on project_name
    # ... (This is where you'll need to implement the actual code generation)
    return "import streamlit as st\nst.title('My Streamlit App')\nst.write('Hello, world!')"

# Function to display the AI Guide chat
def display_ai_guide_chat(chat_history):
    st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
    for user_message, agent_message in chat_history:
        st.markdown(f"<div class='chat-message user'>{user_message}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-message agent'>{agent_message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Terminal", "Explorer", "Code Editor", "Build & Deploy"])

    if app_mode == "Home":
        st.title("Welcome to AI-Guided Development")
        st.write("This application helps you build and deploy applications with the assistance of an AI Guide.")
        st.write("Toggle the AI Guide from the sidebar to choose the level of assistance you need.")

    elif app_mode == "Terminal":
        st.header("Terminal")
        terminal_input = st.text_input("Enter a command:")
        if st.button("Run"):
            output = run_code(terminal_input)
            st.session_state.terminal_history.append((terminal_input, output))
            st.code(output, language="bash")
        if ai_guide_level != "No Assistance":
            st.write("Run commands here to add packages to your project. For example: `pip install <package-name>`.")
            if terminal_input and "install" in terminal_input:
                package_name = terminal_input.split("install")[-1].strip()
                st.write(f"Package `{package_name}` will be added to your project.")

    elif app_mode == "Explorer":
        st.header("Explorer")
        uploaded_file = st.file_uploader("Upload a file", type=["py"])
        if uploaded_file:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
            st.write(file_details)
            save_path = os.path.join(PROJECT_ROOT, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File {uploaded_file.name} saved successfully!")

        st.write("Drag and drop files into the 'app' folder.")
        for project, details in st.session_state.workspace_projects.items():
            st.write(f"Project: {project}")
            for file in details['files']:
                st.write(f"  - {file}")
                if st.button(f"Move {file} to app folder"):
                    # Logic to move file to 'app' folder
                    pass
        if ai_guide_level != "No Assistance":
            st.write("You can upload files and move them into the 'app' folder for building your application.")

    elif app_mode == "Code Editor":
        st.header("Code Editor")
        code_editor = st.text_area("Write your code:", height=300)
        if st.button("Save Code"):
            # Logic to save code
            pass
        if ai_guide_level != "No Assistance":
            st.write("The function `foo()` requires the `bar` package. Add it to `requirements.txt`.")

    elif app_mode == "Build & Deploy":
        st.header("Build & Deploy")
        project_name_input = st.text_input("Enter Project Name for Automation:")
        if st.button("Automate"):
            selected_agent = st.selectbox("Select an AI agent", st.session_state.available_agents)
            selected_model = st.selectbox("Select a code-generative model", AVAILABLE_CODE_GENERATIVE_MODELS)
            agent = AIAgent(selected_agent, "", [])  # Load the agent without skills for now
            summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects, project_name_input, selected_model, hf_token)
            st.write("Autonomous Build Summary:")
            st.write(summary)
            st.write("Next Step:")
            st.write(next_step)
            if agent._hf_api and agent.has_valid_hf_token():
                repository = agent.deploy_built_space_to_hf()
                st.markdown("## Congratulations! Successfully deployed Space 🚀 ##")
                st.markdown("[Check out your new Space here](hf.co/" + repository.name + ")")

    # AI Guide Chat
    if ai_guide_level != "No Assistance":
        display_ai_guide_chat(st.session_state.chat_history)
        # Add a text input for user to interact with the AI Guide
        user_input = st.text_input("Ask the AI Guide a question:", key="user_input")
        if st.button("Send"):
            if user_input:
                # Process the user's input and get a response from the AI Guide
                agent_response = process_input(user_input)
                st.session_state.chat_history.append((user_input, agent_response))
                # Clear the user input field
                st.session_state.user_input = ""

    # CSS for styling
    st.markdown("""
    <style>
    /* Advanced and Accommodating CSS */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f4f4f9;
        color: #333;
        margin: 0;
        padding: 0;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #333;
    }

    .container {
        width: 90%;
        margin: 0 auto;
        padding: 20px;
    }

    /* Navigation Sidebar */
    .sidebar {
        background-color: #2c3e50;
        color: #ecf0f1;
        padding: 20px;
        height: 100vh;
        position: fixed;
        top: 0;
        left: 0;
        width: 250px;
        overflow-y: auto;
    }

    .sidebar a {
        color: #ecf0f1;
        text-decoration: none;
        display: block;
        padding: 10px 0;
    }

    .sidebar a:hover {
        background-color: #34495e;
        border-radius: 5px;
    }

    /* Main Content */
    .main-content {
        margin-left: 270px;
        padding: 20px;
    }

    /* Buttons */
    button {
        background-color: #3498db;
        color: #fff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }

    button:hover {
        background-color: #2980b9;
    }

    /* Text Areas and Inputs */
    textarea, input[type="text"] {
        width: 100%;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #ddd;
        border-radius: 5px;
        box-sizing: border-box;
    }

    textarea:focus, input[type="text"]:focus {
        border-color: #3498db;
        outline: none;
    }

    /* Terminal Output */
    .code-output {
        background-color: #1e1e1e;
        color: #dcdcdc;
        padding: 20px;
        border-radius: 5px;
        font-family: 'Courier New', Courier, monospace;
    }

    /* Chat History */
    .chat-history {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 5px;
        max-height: 300px;
        overflow-y: auto;
    }

    .chat-message {
        margin-bottom: 10px;
    }

    .chat-message.user {
        text-align: right;
        color: #3498db;
    }

    .chat-message.agent {
        text-align: left;
        color: #e74c3c;
    }

    /* Project Management */
    .project-list {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 5px;
        max-height: 300px;
        overflow-y: auto;
    }

    .project-item {
        margin-bottom: 10px;
    }

    .project-item a {
        color: #3498db;
        text-decoration: none;
    }

    .project-item a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)