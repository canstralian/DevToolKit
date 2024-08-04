import json
import time
from typing import Dict, List, Tuple

import gradio as gr
import streamlit as st
import streamlit_chat
from huggingface_hub import InferenceClient, hf_hub_url, cached_download
import git
from langchain_community.llms import HuggingFaceHub
from langchain_community.chains import ConversationChain
from langchain_community.memory import ConversationBufferMemory
from langchain_community.chains.question_answering import load_qa_chain
from langchain_community.utils import CharacterTextSplitter
from transformers import BertTokenizerFast

# --- Constants ---
MODEL_NAME = "google/flan-t5-xl"  # Consider using a more powerful model like 'google/flan-t5-xl'
MAX_NEW_TOKENS = 2048  # Increased for better code generation
TEMPERATURE = 0.7
TOP_P = 0.95
REPETITION_PENALTY = 1.2

# --- Model & Tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")  # Use 'auto' for optimal device selection
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
model_path = os.path.join(os.getcwd(), PRETRAINED_MODEL_NAME)
if not os.path.exists(model_path):
    raise FileNotFoundError("Pre-trained model weight directory {} doesn't exist".format(model_path))
else:
    print("Found Pre-trained Model at:", model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# Download the DistilBERT tokenizer (~3 MB)
DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased').save_pretrained('./cache/distilbert-base-uncased-local')

# --- Agents ---
agents = {
    "WEB_DEV": {
        "description": "Expert in web development technologies and frameworks.",
        "skills": ["HTML", "CSS", "JavaScript", "React", "Vue.js", "Flask", "Django", "Node.js", "Express.js"],
        "system_prompt": "You are a web development expert. Your goal is to assist the user in building and deploying web applications. Provide code snippets, explanations, and guidance on best practices.",
    },
    "AI_SYSTEM_PROMPT": {
        "description": "Expert in designing and implementing AI systems.",
        "skills": ["Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Reinforcement Learning"],
        "system_prompt": "You are an AI system expert. Your goal is to assist the user in designing and implementing AI systems. Provide code snippets, explanations, and guidance on best practices.",
    },
    "PYTHON_CODE_DEV": {
        "description": "Expert in Python programming and development.",
        "skills": ["Python", "Data Structures", "Algorithms", "Object-Oriented Programming", "Functional Programming"],
        "system_prompt": "You are a Python code development expert. Your goal is to assist the user in writing and debugging Python code. Provide code snippets, explanations, and guidance on best practices.",
    },
    "CODE_REVIEW_ASSISTANT": {
        "description": "Expert in code review and quality assurance.",
        "skills": ["Code Style", "Best Practices", "Security", "Performance", "Maintainability"],
        "system_prompt": "You are a code review expert. Your goal is to assist the user in reviewing and improving their code. Provide feedback on code quality, style, and best practices.",
    },
}

# --- Session State ---
if "workspace_projects" not in st.session_state:
    st.session_state.workspace_projects = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "selected_agents" not in st.session_state:
    st.session_state.selected_agents = []
if "current_project" not in st.session_state:
    st.session_state.current_project = None

# --- Helper Functions ---
def add_code_to_workspace(project_name: str, code: str, file_name: str):
    if project_name in st.session_state.workspace_projects:
        st.session_state.workspace_projects[project_name]['files'].append({'file_name': file_name, 'code': code})
        return f"Added code to {file_name} in project {project_name}"
    else:
        return f"Project {project_name} does not exist"

def terminal_interface(command: str, project_name: str):
    if project_name in st.session_state.workspace_projects:
        result = subprocess.run(command, cwd=project_name, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    else:
        return f"Project {project_name} does not exist"

def get_agent_response(message: str, system_prompt: str):
    llm = HuggingFaceHub(repo_id=MODEL_NAME, model_kwargs={"temperature": TEMPERATURE, "top_p": TOP_P, "repetition_penalty": REPETITION_PENALTY, "max_length": MAX_NEW_TOKENS})
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
    response = conversation.run(system_prompt + "\n" + message)
    return response

def display_agent_info(agent_name: str):
    agent = agents[agent_name]
    st.sidebar.subheader(f"Active Agent: {agent_name}")
    st.sidebar.write(f"Description: {agent['description']}")
    st.sidebar.write(f"Skills: {', '.join(agent['skills'])}")

def display_workspace_projects():
    st.subheader("Workspace Projects")
    for project_name, project_data in st.session_state.workspace_projects.items():
        with st.expander(project_name):
            for file in project_data['files']:
                st.text(file['file_name'])
                st.code(file['code'], language="python")

def display_chat_history():
    st.subheader("Chat History")
    html_string = ""
    for idx, message in enumerate(st.session_state.chat_history):
        if idx % 2 == 0:
           role = "User:"
        else:
           role = "Assistant:"
        html_string += f"<p>{role}</p>"
        html_string += f"<p>{message}</p>"
    st.markdown(html_string, unsafe_allow_html=True)

def run_autonomous_build(selected_agents: List[str], project_name: str):
    st.info("Starting autonomous build process...")
    for agent in selected_agents:
        st.write(f"Agent {agent} is working on the project...")
        code = get_agent_response(f"Generate code for a simple web application in project {project_name}", agents[agent]['system_prompt'])
        add_code_to_workspace(project_name, code, f"{agent.lower()}_app.py")
        st.write(f"Agent {agent} has completed its task.")
    st.success("Autonomous build process completed!")

def collaborative_agent_example(selected_agents: List[str], project_name: str, task: str):
    st.info(f"Starting collaborative task: {task}")
    responses = {}
    for agent in selected_agents:
        st.write(f"Agent {agent} is working on the task...")
        response = get_agent_response(task, agents[agent]['system_prompt'])
        responses[agent] = response
    
    combined_response = combine_and_process_responses(responses, task)
    st.success("Collaborative task completed!")
    st.write(combined_response)

def combine_and_process_responses(responses: Dict[str, str], task: str) -> str:
    # This is a placeholder function. In a real-world scenario, you would implement
    # more sophisticated logic to combine and process the responses.
    combined = "\n\n".join([f"{agent}: {response}" for agent, response in responses.items()])
    return f"Combined response for task '{task}':\n\n{combined}"

# --- Streamlit UI ---
st.title("DevToolKit: AI-Powered Development Environment")

# --- Project Management ---
st.header("Project Management")
project_name = st.text_input("Enter project name:")
if st.button("Create Project"):
    if project_name and project_name not in st.session_state.workspace_projects:
        st.session_state.workspace_projects[project_name] = {'files': []}
        st.success(f"Created project: {project_name}")
    elif project_name in st.session_state.workspace_projects:
        st.warning(f"Project {project_name} already exists")
    else:
        st.warning("Please enter a project name")

# --- Code Editor ---
st.subheader("Code Editor")
if st.session_state.workspace_projects:
    selected_project = st.selectbox("Select project", list(st.session_state.workspace_projects.keys()))
    if selected_project:
        files = [file['file_name'] for file in st.session_state.workspace_projects[selected_project]['files']]
        selected_file = st.selectbox("Select file to edit", files) if files else None
        if selected_file:
            file_content = next((file['code'] for file in st.session_state.workspace_projects[selected_project]['files'] if file['file_name'] == selected_file), "")
            edited_code = st_ace(value=file_content, language="python", theme="monokai", key="code_editor")
            if st.button("Save Changes"):
                for file in st.session_state.workspace_projects[selected_project]['files']:
                    if file['file_name'] == selected_file:
                        file['code'] = edited_code
                        st.success("Changes saved successfully!")
                        break
        else:
            st.info("No files in the project. Use the chat interface to generate code.")
else:
    st.info("No projects created yet. Create a project to start coding.")

# --- Terminal Interface ---
st.subheader("Terminal (Workspace Context)")
if st.session_state.workspace_projects:
    selected_project = st.selectbox("Select project for terminal", list(st.session_state.workspace_projects.keys()))
    terminal_input = st.text_input("Enter a command within the workspace:")
    if st.button("Run Command"):
        terminal_output = terminal_interface(terminal_input, selected_project)
        st.code(terminal_output, language="bash")
else:
    st.info("No projects created yet. Create a project to use the terminal.")

# --- Chat Interface ---
st.subheader("Chat with AI Agents")
selected_agents = st.multiselect("Select AI agents", list(agents.keys()), key="agent_select")
st.session_state.selected_agents = selected_agents
agent_chat_input = st.text_area("Enter your message for the agents:", key="agent_input")
if st.button("Send to Agents", key="agent_send"):
    if selected_agents and agent_chat_input:
        responses = {}
        for agent in selected_agents:
            response = get_agent_response(agent_chat_input, agents[agent]['system_prompt'])
            responses[agent] = response
        st.session_state.chat_history.append(f"User: {agent_chat_input}")
        for agent, response in responses.items():
            st.session_state.chat_history.append(f"{agent}: {response}")
        st_chat(st.session_state.chat_history)  # Display chat history using st_chat
    else:
        st.warning("Please select at least one agent and enter a message.")

# --- Agent Control ---
st.subheader("Agent Control")
for agent_name in agents:
    agent = agents[agent_name]
    with st.expander(f"{agent_name} ({agent['description']})"):
        if st.button(f"Activate {agent_name}", key=f"activate_{agent_name}"):
            st.session_state.active_agent = agent_name
            st.success(f"{agent_name} activated.")
        if st.button(f"Deactivate {agent_name}", key=f"deactivate_{agent_name}"):
            st.session_state.active_agent = None
            st.success(f"{agent_name} deactivated.")

# --- Automate Build Process ---
st.subheader("Automate Build Process")
if st.button("Automate"):
    if st.session_state.selected_agents and project_name:
        run_autonomous_build(st.session_state.selected_agents, project_name)
    else:
        st.warning("Please select at least one agent and create a project.")

# --- Version Control ---
st.subheader("Version Control")
repo_url = st.text_input("Enter repository URL:")
if st.button("Clone Repository"):
    if repo_url and project_name:
        try:
            git.Repo.clone_from(repo_url, project_name)
            st.success(f"Repository cloned successfully to {project_name}")
        except git.GitCommandError as e:
            st.error(f"Error cloning repository: {e}")
    else:
        st.warning("Please enter a repository URL and create a project.")

# --- Collaborative Agent Example ---
st.subheader("Collaborative Agent Example")
collab_agents = st.multiselect("Select AI agents for collaboration", list(agents.keys()), key="collab_agent_select")
collab_project = st.text_input("Enter project name for collaboration:")
collab_task = st.text_input("Enter a task for the agents to collaborate on:")
if st.button("Run Collaborative Task"):
    if collab_agents and collab_project and collab_task:
        collaborative_agent_example(collab_agents, collab_project, collab_task)
    else:
        st.warning("Please select agents, enter a project name, and specify a task.")

# --- Display Information ---
st.sidebar.subheader("Current State")
st.sidebar.json(st.session_state)
if st.session_state.active_agent:
    display_agent_info(st.session_state.active_agent)
display_workspace_projects()
display_chat_history()

if __name__ == "__main__":
    st.sidebar.title("DevToolKit")
    st.sidebar.info("This is an AI-powered development environment.")