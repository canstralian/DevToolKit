import os
import json
import time
from typing import Dict, List, Tuple

import gradio as gr
import streamlit as st
from huggingface_hub import InferenceClient, hf_hub_url, cached_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from rich import print as rprint
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
import subprocess
import threading

# --- Constants ---
MODEL_NAME = "bigscience/bloom-1b7"  # Choose a suitable model
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.95
REPETITION_PENALTY = 1.2

# --- Model & Tokenizer ---
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
        "system_prompt": "You are a code review assistant. Your goal is to assist the user in reviewing code for quality and efficiency. Provide feedback on code style, best practices, security, performance, and maintainability.",
    },
    "CONTENT_WRITER_EDITOR": {
        "description": "Expert in content writing and editing.",
        "skills": ["Grammar", "Style", "Clarity", "Conciseness", "SEO"],
        "system_prompt": "You are a content writer and editor. Your goal is to assist the user in creating high-quality content. Provide suggestions on grammar, style, clarity, conciseness, and SEO.",
    },
    "QUESTION_GENERATOR": {
        "description": "Expert in generating questions for learning and assessment.",
        "skills": ["Question Types", "Cognitive Levels", "Assessment Design"],
        "system_prompt": "You are a question generator. Your goal is to assist the user in generating questions for learning and assessment. Provide questions that are relevant to the topic and aligned with the cognitive levels.",
    },
    "HUGGINGFACE_FILE_DEV": {
        "description": "Expert in developing Hugging Face files for machine learning models.",
        "skills": ["Transformers", "Datasets", "Model Training", "Model Deployment"],
        "system_prompt": "You are a Hugging Face file development expert. Your goal is to assist the user in creating and deploying Hugging Face files for machine learning models. Provide code snippets, explanations, and guidance on best practices.",
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
if "current_agent" not in st.session_state:
    st.session_state.current_agent = None
if "current_cluster" not in st.session_state:
    st.session_state.current_cluster = None
if "hf_token" not in st.session_state:
    st.session_state.hf_token = None
if "repo_name" not in st.session_state:
    st.session_state.repo_name = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

def add_code_to_workspace(project_name: str, code: str, file_name: str) -> str:
    if project_name in st.session_state.workspace_projects:
        project = st.session_state.workspace_projects[project_name]
        project['files'].append({'file_name': file_name, 'code': code})
        return f"Code added to project {project_name}"
    else:
        return f"Project {project_name} does not exist."

def terminal_interface(command: str, project_name: str) -> str:
    try:
        project = st.session_state.workspace_projects.get(project_name, {})
        workspace_dir = os.path.join("workspace", project_name)
        os.makedirs(workspace_dir, exist_ok=True)
        result = subprocess.run(command, shell=True, cwd=workspace_dir, capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

def chat_interface(message: str, selected_agents: List[str]) -> str:
    responses = {}
    for agent_name in selected_agents:
        agent = agents[agent_name]
        responses[agent_name] = agent['system_prompt'] + " " + message
    return json.dumps(responses, indent=2)

def run_autonomous_build(selected_agents: List[str], project_name: str):
    for agent_name in selected_agents:
        agent = agents[agent_name]
        chat_history = st.session_state.chat_history
        workspace_projects = st.session_state.workspace_projects
        summary, next_step = agent.autonomous_build(chat_history, workspace_projects)
        rprint(Panel(summary, title="[bold blue]Current State[/bold blue]"))
        rprint(Panel(next_step, title="[bold blue]Next Step[/bold blue]"))
        # Implement logic for autonomous build based on the current state

def display_agent_info(agent_name: str):
    agent = agents[agent_name]
    st.sidebar.subheader(f"Agent: {agent_name}")
    st.sidebar.write(agent['description'])
    st.sidebar.write("Skills: " + ", ".join(agent['skills']))
    st.sidebar.write("System Prompt: " + agent['system_prompt'])

def display_workspace_projects():
    st.sidebar.subheader("Workspace Projects")
    for project_name, details in st.session_state.workspace_projects.items():
        st.sidebar.write(f"{project_name}: {details}")

def display_chat_history():
    st.sidebar.subheader("Chat History")
    st.sidebar.json(st.session_state.chat_history)

# --- Streamlit UI ---
st.title("DevToolKit: AI-Powered Development Environment")

# --- Project Management ---
st.header("Project Management")
project_name = st.text_input("Enter project name:")
if st.button("Create Project"):
    if project_name not in st.session_state.workspace_projects:
        st.session_state.workspace_projects[project_name] = {'files': []}
        st.success(f"Created project: {project_name}")
    else:
        st.warning(f"Project {project_name} already exists")

# --- Code Addition ---
st.subheader("Add Code to Workspace")
code_to_add = st.text_area("Enter code to add to workspace:")
file_name = st.text_input("Enter file name (e.g. 'app.py'):")
if st.button("Add Code"):
    add_code_status = add_code_to_workspace(project_name, code_to_add, file_name)
    st.success(add_code_status)

# --- Terminal Interface ---
st.subheader("Terminal (Workspace Context)")
terminal_input = st.text_input("Enter a command within the workspace:")
if st.button("Run Command"):
    terminal_output = terminal_interface(terminal_input, project_name)
    st.code(terminal_output, language="bash")

# --- Chat Interface ---
st.subheader("Chat with AI Agents")
selected_agents = st.multiselect("Select AI agents", list(agents.keys()), key="agent_select")
st.session_state.selected_agents = selected_agents
agent_chat_input = st.text_area("Enter your message for the agents:", key="agent_input")
if st.button("Send to Agents", key="agent_send"):
    agent_chat_response = chat_interface(agent_chat_input, selected_agents)
    st.write(agent_chat_response)

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
    if st.session_state.selected_agents:
        run_autonomous_build(st.session_state.selected_agents, project_name)
    else:
        st.warning("Please select at least one agent.")

# --- Display Information ---
st.sidebar.subheader("Current State")
st.sidebar.json(st.session_state, indent=2)
if st.session_state.active_agent:
    display_agent_info(st.session_state.active_agent)
display_workspace_projects()
display_chat_history()

# --- Gradio Interface ---
additional_inputs = [
    gr.Dropdown(label="Agents", choices=[s for s in agents.keys()], value=list(agents.keys())[0], interactive=True),
    gr.Textbox(label="System Prompt", max_lines=1, interactive=True),
    gr.Slider(label="Temperature", value=TEMPERATURE, minimum=0.0, maximum=1.0, step=0.05, interactive=True, info="Higher values produce more diverse outputs"),
    gr.Slider(label="Max new tokens", value=MAX_NEW_TOKENS, minimum=0, maximum=10240, step=64, interactive=True, info="The maximum numbers of new tokens"),
    gr.Slider(label="Top-p (nucleus sampling)", value=TOP_P, minimum=0.0, maximum=1, step=0.05, interactive=True, info="Higher values sample more low-probability tokens"),
    gr.Slider(label="Repetition penalty", value=REPETITION_PENALTY, minimum=1.0, maximum=2.0, step=0.05, interactive=True, info="Penalize repeated tokens"),
]

examples = [
    ["Create a simple web application using Flask", "WEB_DEV"],
    ["Generate a Python script to perform a linear regression analysis", "PYTHON_CODE_DEV"],
    ["Create a Dockerfile for a Node.js application", "AI_SYSTEM_PROMPT"],
    # Add more examples as needed
]

gr.ChatInterface(
    fn=chat_interface,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=additional_inputs,
    title="DevToolKit AI Assistant",
    examples=examples,
    concurrency_limit=20,
).launch(show_api=True)