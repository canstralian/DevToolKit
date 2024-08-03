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
if "selected_code_model" not in st.session_state:
    st.session_state.selected_code_model = None
if "selected_chat_model" not in st.session_state:
    st.session_state.selected_chat_model = None

# --- Functions ---
def format_prompt(message: str, history: List[Tuple[str, str]], agent_prompt: str) -> str:
    """Formats the prompt for the language model."""
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {agent_prompt}, {message} [/INST]"
    return prompt

def generate_response(prompt: str, agent_name: str) -> str:
    """Generates a response from the language model."""
    agent = agents[agent_name]
    system_prompt = agent["system_prompt"]
    generate_kwargs = dict(
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, **generate_kwargs)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def chat_interface(chat_input: str, agent_names: List[str]) -> str:
    """Handles chat interactions with the selected agents."""
    if agent_names:
        responses = []
        for agent_name in agent_names:
            prompt = format_prompt(chat_input, st.session_state.chat_history, agents[agent_name]["system_prompt"])
            response = generate_response(prompt, agent_name)
            responses.append(f"{agent_name}: {response}")
        return "\n".join(responses)
    else:
        return "Please select at least one agent."

def terminal_interface(command: str, project_name: str) -> str:
    """Executes a command within the specified project directory."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=project_name)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return str(e)

def add_code_to_workspace(project_name: str, code: str, file_name: str) -> str:
    """Adds code to a workspace project."""
    project_path = os.path.join(os.getcwd(), project_name)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    file_path = os.path.join(project_path, file_name)
    with open(file_path, 'w') as file:
        file.write(code)
    if project_name not in st.session_state.workspace_projects:
        st.session_state.workspace_projects[project_name] = {'files': []}
    st.session_state.workspace_projects[project_name]['files'].append(file_name)
    return f"Added {file_name} to {project_name}"

def display_workspace_projects():
    """Displays a table of workspace projects."""
    table = Table(title="Workspace Projects")
    table.add_column("Project Name", style="cyan", no_wrap=True)
    table.add_column("Files", style="magenta")
    for project_name, details in st.session_state.workspace_projects.items():
        table.add_row(project_name, ", ".join(details['files']))
    rprint(Panel(table, title="[bold blue]Workspace Projects[/bold blue]"))

def display_chat_history():
    """Displays the chat history in a formatted way."""
    table = Table(title="Chat History")
    table.add_column("User", style="cyan", no_wrap=True)
    table.add_column("Agent", style="magenta")
    for user_prompt, bot_response in st.session_state.chat_history:
        table.add_row(user_prompt, bot_response)
    rprint(Panel(table, title="[bold blue]Chat History[/bold blue]"))

def display_agent_info(agent_name: str):
    """Displays information about the selected agent."""
    agent = agents[agent_name]
    table = Table(title=f"{agent_name} - Agent Information")
    table.add_column("Description", style="cyan", no_wrap=True)
    table.add_column("Skills", style="magenta")
    table.add_row(agent["description"], ", ".join(agent["skills"]))
    rprint(Panel(table, title=f"[bold blue]{agent_name} - Agent Information[/bold blue]"))

def run_autonomous_build(agent_names: List[str], project_name: str):
    """Runs the autonomous build process."""
    for agent_name in agent_names:
        agent = agents[agent_name]
        chat_history = st.session_state.chat_history
        workspace_projects = st.session_state.workspace_projects
        summary, next_step = agent.autonomous_build(chat_history, workspace_projects)
        rprint(Panel(summary, title="[bold blue]Current State[/bold blue]"))
        rprint(Panel(next_step, title="[bold blue]Next Step[/bold blue]"))
        # Implement logic for autonomous build based on the current state
        # ...

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
st.sidebar.json(st.session_state.current_state)
if st.session_state.active_agent:
    display_agent_info(st.session_state.active_agent)
display_workspace_projects()
display_chat_history()

# --- Gradio Interface ---
additional_inputs = [
    gr.Dropdown(label="Agents", choices=[s for s in agents.keys()], value=list(agents.keys())[0], interactive=True),
    gr.Textbox(label="System Prompt", max_lines=1, interactive=True),
    gr.Slider(label="Temperature", value=TEMPERATURE, minimum=0.0, maximum=1.0, step=0.05, interactive=True, info="Higher values produce more diverse outputs"),
    gr.Slider(label="Max new tokens", value=MAX_NEW_TOKENS, minimum=0, maximum=1000*10, step=64, interactive=True, info="The maximum numbers of new tokens"),
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