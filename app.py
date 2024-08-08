import streamlit as st
import os
import subprocess
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import json

try:
    huggingface_json = st.secrets["huggingface"]
    huggingface_data = json.loads(huggingface_json)
    huggingface_token = huggingface_data["hf_token"]
except Exception as e:
    st.error(f"Unable to load secrets: {str(e)}")
    raise

PROJECT_ROOT = "home/app/projects"
AGENT_DIRECTORY = "home/app/agents"
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

# AI Guide Toggle
ai_guide_level = st.sidebar.radio("AI Guide Level", ["Full Assistance", "Partial Assistance", "No Assistance"])

class AIAgent:
    def __init__(self, name, description, skills):
        self.name = name
        self.description = description
        self.skills = skills
        self._hf_api = HfApi(token=huggingface_token)

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
        # Implement logic to deploy the built space to Hugging Face Spaces
        # This could involve creating a new Space, uploading files, and configuring the Space
        # Use the HfApi to interact with the Hugging Face API
        # Example:
        # repository = self._hf_api.create_repo(repo_id="my-new-space", private=False)
        # ... upload files to the repository
        # ... configure the Space
        # return repository
        pass

    def has_valid_hf_token(self):
        return self._hf_api.whoami() is not None

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

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="AI-Powered Development Platform")

    # Sidebar
    st.sidebar.title("AI Development Platform")
    st.sidebar.header("Tool Box")
    st.sidebar.subheader("Workspace Management")
    project_name_input = st.sidebar.text_input("Project Name:")
    create_project_button = st.sidebar.button("Create Project")
    if create_project_button:
        st.sidebar.write(workspace_interface(project_name_input))

    st.sidebar.subheader("Code Generation")
    selected_model = st.sidebar.selectbox("Select Code Model", AVAILABLE_CODE_GENERATIVE_MODELS)
    code_input = st.sidebar.text_area("Enter Code Prompt:")
    generate_code_button = st.sidebar.button("Generate Code")
    if generate_code_button:
        if selected_model:
            tokenizer = AutoTokenizer.from_pretrained(selected_model)
            model = AutoModelForCausalLM.from_pretrained(selected_model)
            code_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            generated_code = code_generator(code_input, max_length=500, num_return_sequences=1)[0]['generated_text']
            st.sidebar.code(generated_code)
        else:
            st.sidebar.error("Please select a code model.")

    st.sidebar.subheader("Terminal")
    command = st.sidebar.text_input("Enter a command:")
    if command:
        output, error = run_command(command)
        if error:
            st.sidebar.error(f"Error executing command: {error}")
        else:
            st.sidebar.code(output)

    # Main Content
    st.title("AI-Powered Development Platform")
    st.header("Workspace")
    st.subheader("Chat with an AI Agent")
    chat_input = st.text_input("Enter your message:")
    if chat_input:
        st.session_state.chat_history.append((chat_input, process_input(chat_input)))

    st.markdown("## Chat History ##")
    st.markdown(display_chat_history(st.session_state.chat_history))

    st.subheader("Available Agents")
    for agent_name in st.session_state.available_agents:
        st.write(f"**{agent_name}**")

    st.subheader("Project Management")
    st.markdown(display_workspace_projects(st.session_state.workspace_projects))

    # AI Guide
    if ai_guide_level == "Full Assistance":
        st.markdown("## AI Guide: Full Assistance ##")
        st.write("**Recommended Action:**")
        st.write("Create a new project and then generate some code.")
    elif ai_guide_level == "Partial Assistance":
        st.markdown("## AI Guide: Partial Assistance ##")
        st.write("**Tips:**")
        st.write("Use the chat interface to ask questions about your project.")
        st.write("Use the code generation tool to generate code snippets.")
    else:
        st.markdown("## AI Guide: No Assistance ##")
        st.write("You are on your own!")

    # Autonomous Build
    if st.button("Autonomous Build"):
        project_name = project_name_input
        selected_model = selected_model
        agent = AIAgent("Code Architect", "I am an expert in code generation and deployment.", ["Code Generation", "Deployment"])
        summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects, project_name, selected_model, huggingface_token)
        st.write("Autonomous Build Summary:")
        st.write(summary)
        st.write("Next Step:")
        st.write(next_step)
        if agent._hf_api and agent.has_valid_hf_token():
            repository = agent.deploy_built_space_to_hf()
            st.markdown("## Congratulations! Successfully deployed Space 🚀 ##")
            st.markdown("[Check out your new Space here](hf.co/" + repository.name + ")")

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