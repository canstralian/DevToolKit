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
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqLMForCausalGeneration

def create_causal_lm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).causal_decoder
    return model, tokenizer

AutoModelForCausalLM = lambda model_name: create_causal_lm(model_name)[0]
AutoTokenizerForCausalLM = lambda model_name: create_causal_lm(model_name)[1]


# --- Constants ---
MODEL_NAME = "bigscience/bloom-1b7"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.95
REPETITION_PENALTY = 1.2

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

def chat_interface(message: str, selected_agents: List[str]):
    responses = {}
    for agent in selected_agents:
        responses[agent] = get_agent_response(message, agents[agent]['system_prompt'])
    return responses

def get_agent_response(message: str, system_prompt: str):
    llm = HuggingFaceHub(repo_id=MODEL_NAME, model_kwargs={"temperature": TEMPERATURE, "top_p": TOP_P, "repetition_penalty": REPETITION_PENALTY})
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
    response = conversation.run(system_prompt + "\n" + message)
    return response

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
st.sidebar.json(st.session_state)
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

# --- Helper Functions ---

def display_agent_info(agent_name: str):
    agent = agents[agent_name]
    st.sidebar.subheader(f"Active Agent: {agent_name}")
    st.sidebar.write(f"Description: {agent['description']}")
    st.sidebar.write(f"Skills: {', '.join(agent['skills'])}")

def display_workspace_projects():
    st.sidebar.subheader("Workspace Projects")
    if st.session_state.workspace_projects:
        for project_name in st.session_state.workspace_projects:
            st.sidebar.write(f"- {project_name}")
    else:
        st.sidebar.write("No projects created yet.")

def display_chat_history():
    st.sidebar.subheader("Chat History")
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            st.sidebar.write(message)
    else:
        st.sidebar.write("No chat history yet.")

def run_autonomous_build(selected_agents: List[str], project_name: str):
    # This function should implement the autonomous build process
    # It should use the selected agents and the project name to generate code and run commands
    # You can use the `get_agent_response` function to get responses from agents
    # You can use the `add_code_to_workspace` and `terminal_interface` functions to manage the workspace
    st.write("Running autonomous build...")
    for agent in selected_agents:
        # Example: Get code from the agent
        code = get_agent_response(f"Generate code for a simple web application in project {project_name}", agents[agent]['system_prompt'])
        # Example: Add code to the workspace
        add_code_to_workspace(project_name, code, "app.py")
        # Example: Run a command in the workspace
        terminal_interface("python app.py", project_name)
    st.write("Autonomous build completed.")

# --- Collaborative Agent Example ---

def collaborative_agent_example(selected_agents: List[str], project_name: str, task: str):
    # Example: Collaborative code generation
    st.write(f"Running collaborative task: {task}")
    responses = []
    for agent in selected_agents:
        response = get_agent_response(f"As a {agent}, please contribute to the following task: {task}", agents[agent]['system_prompt'])
        responses.append(response)

    # Combine responses and process them
    combined_response = "\n".join(responses)
    st.write(f"Combined response:\n{combined_response}")

    # Example: Use code review agent for feedback
    if "CODE_REVIEW_ASSISTANT" in selected_agents:
        review_response = get_agent_response(f"Review the following code and provide feedback: {combined_response}", agents["CODE_REVIEW_ASSISTANT"]['system_prompt'])
        st.write(f"Code Review Feedback:\n{review_response}")

    # Example: Use content writer for documentation
    if "CONTENT_WRITER_EDITOR" in selected_agents:
        documentation_response = get_agent_response(f"Generate documentation for the following code: {combined_response}", agents["CONTENT_WRITER_EDITOR"]['system_prompt'])
        st.write(f"Documentation:\n{documentation_response}")

# --- Streamlit UI for Collaborative Agent Example ---

st.subheader("Collaborative Agent Example")
selected_agents_example = st.multiselect("Select AI agents for collaboration", list(agents.keys()), key="agent_select_example")
project_name_example = st.text_input("Enter project name (for example purposes):")
task_example = st.text_input("Enter a task for the agents to collaborate on:")
if st.button("Run Collaborative Task"):
    collaborative_agent_example(selected_agents_example, project_name_example, task_example)
