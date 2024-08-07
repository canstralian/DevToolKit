import os
import subprocess
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel, RagRetriever, AutoModelForSeq2SeqLM
import black
from pylint import lint
from io import StringIO
import sys
import torch
from huggingface_hub import hf_hub_url, cached_download, HfApi
from datetime import datetime

# Set your Hugging Face API key here
hf_token = "YOUR_HUGGING_FACE_API_KEY"  # Replace with your actual token

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

# List of top downloaded free code-generative models from Hugging Face Hub
AVAILABLE_CODE_GENERATIVE_MODELS = [
    "bigcode/starcoder",  # Popular and powerful
    "Salesforce/codegen-350M-mono",  # Smaller, good for quick tasks
    "microsoft/CodeGPT-small",  # Smaller, good for quick tasks
    "google/flan-t5-xl",  # Powerful, good for complex tasks
    "facebook/bart-large-cnn",  # Good for text-to-code tasks
]

# Load pre-trained RAG retriever
rag_retriever = RagRetriever.from_pretrained("facebook/rag-sequence-base")  # Use a Hugging Face RAG model

# Load pre-trained chat model
chat_model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/DialoGPT-medium")  # Use a Hugging Face chat model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

class AIAgent:
    def __init__(self, name, description, skills, hf_api=None):
        self.name = name
        self.description = description
        self.skills = skills
        self._hf_api = hf_api
        self._hf_token = hf_token  # Store the token here

    @property
    def hf_api(self):
        if not self._hf_api and self.has_valid_hf_token():
            self._hf_api = HfApi(token=self._hf_token)
        return self._hf_api

    def has_valid_hf_token(self):
        return bool(self._hf_token)

    async def autonomous_build(self, chat_history, workspace_projects, project_name, selected_model, hf_token):
        self._hf_token = hf_token
        # Continuation of previous methods

    def deploy_built_space_to_hf(self):
        if not self._hf_api or not self._hf_token:
            raise ValueError("Cannot deploy the Space since no valid Hugging Face API connection was established.")
        repository_name = f"my-awesome-space_{datetime.now().timestamp()}"
        files = get_built_space_files()
        commit_response = self.hf_api.commit_repo(
            repo_id=repository_name,
            branch="main",
            commits=[{"message": "Built Space Commit", "tree": tree_payload}]
        )
        print("Commit successful:", commit_response)
        self.publish_space(repository_name)

    def publish_space(self, repository_name):
        publishing_response = self.hf_api.create_model_version(
            model_name=repository_name,
            repo_id=repository_name,
            model_card={},
            library_card={}
        )
        print("Space published:", publishing_response)

def process_input(user_input):
    # Input pipeline: Tokenize and preprocess user input
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids
    attention_mask = tokenizer(user_input, return_tensors="pt").attention_mask

    # RAG model: Generate response
    with torch.no_grad():
        output = rag_retriever(input_ids, attention_mask=attention_mask)
        response = output.generator_outputs[0].sequences[0]

    # Chat model: Refine response
    chat_input = tokenizer(response, return_tensors="pt")
    chat_input["input_ids"] = chat_input["input_ids"].unsqueeze(0)
    chat_input["attention_mask"] = chat_input["attention_mask"].unsqueeze(0)
    with torch.no_grad():
        chat_output = chat_model(**chat_input)
        refined_response = chat_output.sequences[0]

    # Output pipeline: Return final response
    return refined_response

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

def run_code(command, project_name=None):
    if project_name:
        project_path = os.path.join(PROJECT_ROOT, project_name)
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=project_path)
    else:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

def display_chat_history(history):
    chat_history = ""
    for user_input, response in history:
        chat_history += f"User: {user_input}\nAgent: {response}\n\n"
    return chat_history

def display_workspace_projects(projects):
    workspace_projects = ""
    for project, details in projects.items():
        workspace_projects += f"Project: {project}\nFiles:\n"
        for file in details['files']:
            workspace_projects += f"  - {file}\n"
    return workspace_projects

# Streamlit App
st.title("AI Agent Creator")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["AI Agent Creator", "Tool Box", "Workspace Chat App"])

if app_mode == "AI Agent Creator":
    # AI Agent Creator
    st.header("Create an AI Agent from Text")

    st.subheader("From Text")
    agent_name = st.text_input("Enter agent name:")
    text_input = st.text_area("Enter skills (one per line):")
    if st.button("Create Agent"):
        skills = text_input.split('\n')
        agent = AIAgent(agent_name, "AI agent created from text input", skills)
        st.success(f"Agent '{agent_name}' created and saved successfully.")
        st.session_state.available_agents.append(agent_name)

elif app_mode == "Tool Box":
    # Tool Box
    st.header("AI-Powered Tools")

    # Chat Interface
    st.subheader("Chat with CodeCraft")
    chat_input = st.text_area("Enter your message:")
    if st.button("Send"):
        response = process_input(chat_input)
        st.session_state.chat_history.append((chat_input, response))
        st.write(f"CodeCraft: {response}")

    # Terminal Interface
    st.subheader("Terminal")
    terminal_input = st.text_input("Enter a command:")
    if st.button("Run"):
        output = run_code(terminal_input)
        st.session_state.terminal_history.append((terminal_input, output))
        st.code(output, language="bash")

    # Project Management
    st.subheader("Project Management")
    project_name_input = st.text_input("Enter Project Name:")
    if st.button("Create Project"):
        status = workspace_interface(project_name_input)
        st.write(status)

    code_to_add = st.text_area("Enter Code to Add to Workspace:", height=150)
    file_name_input = st.text_input("Enter File Name (e.g., 'app.py'):")
    if st.button("Add Code"):
        status = add_code_to_workspace(project_name_input, code_to_add, file_name_input)
        st.write(status)

    # Display Chat History
    st.subheader("Chat History")
    chat_history = display_chat_history(st.session_state.chat_history)
    st.text_area("Chat History", value=chat_history, height=200)

    # Display Workspace Projects
    st.subheader("Workspace Projects")
    workspace_projects = display_workspace_projects(st.session_state.workspace_projects)
    st.text_area("Workspace Projects", value=workspace_projects, height=200)

elif app_mode == "Workspace Chat App":
    # Workspace Chat App
    st.header("Workspace Chat App")

    # Chat Interface with AI Agents
    st.subheader("Chat with AI Agents")
    selected_agent = st.selectbox("Select an AI agent", st.session_state.available_agents)
    agent_chat_input = st.text_area("Enter your message for the agent:")
    if st.button("Send to Agent"):
        response = process_input(agent_chat_input)
        st.session_state.chat_history.append((agent_chat_input, response))
        st.write(f"{selected_agent}: {response}")

    # Code Generation
    st.subheader("Code Generation")
    code_idea = st.text_input("Enter your code idea:")
    selected_model = st.selectbox("Select a code-generative model", AVAILABLE_CODE_GENERATIVE_MODELS)
    if st.button("Generate Code"):
        generated_code = run_code(code_idea)
        st.code(generated_code, language="python")

    # Automate Build Process
    st.subheader("Automate Build Process")
    if st.button("Automate"):
        agent = AIAgent(selected_agent, "", [])  # Load the agent without skills for now
        summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects, project_name, selected_model, hf_token)
        st.write("Autonomous Build Summary:")
        st.write(summary)
        st.write("Next Step:")
        st.write(next_step)

        if agent._hf_api and agent.has_valid_hf_token():
            repository = agent.deploy_built_space_to_hf()
            st.markdown("## Congratulations! Successfully deployed Space 🚀 ##")
            st.markdown("[Check out your new Space here](hf.co/" + repository.name + ")")

    if __name__ == "__main__":
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.selectbox("Choose the app mode", ["AI Agent Creator", "Tool Box", "Workspace Chat App"])

    if app_mode == "AI Agent Creator":
        # AI Agent Creator
        st.header("Create an AI Agent from Text")

        st.subheader("From Text")
        agent_name = st.text_input("Enter agent name:")
        text_input = st.text_area("Enter skills (one per line):")
        if st.button("Create Agent"):
            skills = text_input.split('\n')
            agent = AIAgent(agent_name, "AI agent created from text input", skills)
            st.success(f"Agent '{agent_name}' created and saved successfully.")
            st.session_state.available_agents.append(agent_name)

    elif app_mode == "Tool Box":
        # Tool Box
        st.header("AI-Powered Tools")

        # Chat Interface
        st.subheader("Chat with CodeCraft")
        chat_input = st.text_area("Enter your message:")
        if st.button("Send"):
            response = process_input(chat_input)
            st.session_state.chat_history.append((chat_input, response))
            st.write(f"CodeCraft: {response}")

        # Terminal Interface
        st.subheader("Terminal")
        terminal_input = st.text_input("Enter a command:")
        if st.button("Run"):
            output = run_code(terminal_input)
            st.session_state.terminal_history.append((terminal_input, output))
            st.code(output, language="bash")

        # Project Management
        st.subheader("Project Management")
        project_name_input = st.text_input("Enter Project Name:")
        if st.button("Create Project"):
            status = workspace_interface(project_name_input)
            st.write(status)

        code_to_add = st.text_area("Enter Code to Add to Workspace:", height=150)
        file_name_input = st.text_input("Enter File Name (e.g., 'app.py'):")
        if st.button("Add Code"):
            status = add_code_to_workspace(project_name_input, code_to_add, file_name_input)
            st.write(status)

        # Display Chat History
        st.subheader("Chat History")
        chat_history = display_chat_history(st.session_state.chat_history)
        st.text_area("Chat History", value=chat_history, height=200)

        # Display Workspace Projects
        st.subheader("Workspace Projects")
        workspace_projects = display_workspace_projects(st.session_state.workspace_projects)
        st.text_area("Workspace Projects", value=workspace_projects, height=200)

    elif app_mode == "Workspace Chat App":
        # Workspace Chat App
        st.header("Workspace Chat App")

        # Chat Interface with AI Agents
        st.subheader("Chat with AI Agents")
        selected_agent = st.selectbox("Select an AI agent", st.session_state.available_agents)
        agent_chat_input = st.text_area("Enter your message for the agent:")
        if st.button("Send to Agent"):
            response = process_input(agent_chat_input)
            st.session_state.chat_history.append((agent_chat_input, response))
            st.write(f"{selected_agent}: {response}")

        # Code Generation
        st.subheader("Code Generation")
        code_idea = st.text_input("Enter your code idea:")
        selected_model = st.selectbox("Select a code-generative model", AVAILABLE_CODE_GENERATIVE_MODELS)
        if st.button("Generate Code"):
            generated_code = run_code(code_idea)
            st.code(generated_code, language="python")

        # Automate Build Process
        st.subheader("Automate Build Process")
        if st.button("Automate"):
            agent = AIAgent(selected_agent, "", [])  # Load the agent without skills for now
            summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects, project_name, selected_model, hf_token)
            st.write("Autonomous Build Summary:")
            st.write(summary)
            st.write("Next Step:")
            st.write(next_step)

            if agent._hf_api and agent.has_valid_hf_token():
                repository = agent.deploy_built_space_to_hf()
                st.markdown("## Congratulations! Successfully deployed Space 🚀 ##")
                st.markdown("[Check out your new Space here](hf.co/" + repository.name + ")")

# Launch the Streamlit app
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