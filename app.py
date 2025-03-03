import os
import subprocess
import sys
import streamlit as st
import black
from pylint import lint
from io import StringIO
import logging
import atexit
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

HUGGING_FACE_REPO_URL = "https://huggingface.co/spaces/acecalisto3/DevToolKit"
PROJECT_ROOT = "projects"
AGENT_DIRECTORY = "agents"

# Initialize session state
def initialize_session_state():
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

initialize_session_state()

class InstructModel:
    def __init__(self):
        """Initialize the Mixtral-8x7B-Instruct model"""
        self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            raise EnvironmentError(f"Failed to load model: {str(e)}")

    def generate_response(self, prompt: str) -> str:
        """Generate a response using the Mixtral model"""
        formatted_prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.clean_response(outputs, formatted_prompt)

    def format_prompt(self, prompt: str) -> str:
        """Format the prompt for the model"""
        return f"<s>[INST] {prompt} [/INST]"

    def clean_response(self, outputs, formatted_prompt: str) -> str:
        """Decode and clean up the model's response"""
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(formatted_prompt, "").strip()

    def __del__(self):
        """Cleanup when the model is no longer needed"""
        try:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
        except:
            pass

class AIAgent:
    def __init__(self, name, description, skills):
        self.name = name
        self.description = description
        self.skills = skills

    def create_agent_prompt(self):
        skills_str = '\n'.join([f"* {skill}" for skill in self.skills])
        return f"""
        As an elite expert developer, my name is {self.name}. I possess a comprehensive understanding of the following areas:
        {skills_str}
        I am confident that I can leverage my expertise to assist you in developing and deploying cutting-edge web applications. Please feel free to ask any questions or present any challenges you may encounter.
        """

    def autonomous_build(self, chat_history, workspace_projects):
        summary = self.summarize_chat_history(chat_history)
        summary += "\n\nWorkspace Projects:\n" + self.summarize_workspace_projects(workspace_projects)
        next_step = "Based on the current state, the next logical step is to implement the main application logic."
        return summary, next_step

    def summarize_chat_history(self, chat_history):
        return "Chat History:\n" + "\n".join([f":User  {u}\nAgent: {a}" for u, a in chat_history])

    def summarize_workspace_projects(self, workspace_projects):
        return "\n".join([f"{p}: {details}" for p, details in workspace_projects.items()])

def save_agent_to_file(agent):
    os.makedirs(AGENT_DIRECTORY, exist_ok=True)
    file_path = os.path.join(AGENT_DIRECTORY, f"{agent.name}.txt")
    config_path = os.path.join(AGENT_DIRECTORY, f"{agent.name}Config.txt")
    with open(file_path, "w") as file:
        file.write(agent.create_agent_prompt())
    with open(config_path, "w") as file:
        file.write(f"Agent Name: {agent.name}\nDescription: {agent.description}")
    st.session_state.available_agents.append(agent.name)
    commit_and_push_changes(f"Add agent {agent.name}")

def load_agent_prompt(agent_name):
    file_path = os.path.join(AGENT_DIRECTORY, f"{agent_name}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read()
    return None

def create_agent_from_text(name, text):
    skills = text.split('\n')
    agent = AIAgent(name, "AI agent created from text input.", skills)
    save_agent_to_file(agent)
    return agent.create_agent_prompt()

def chat_interface(input_text):
    """Handles chat interactions without a specific agent."""
    model = InstructModel()
    return model.generate_response(f":User  {input_text}\nAI:")

def chat_interface_with_agent(input_text, agent_name):
    agent_prompt = load_agent_prompt(agent_name)
    if agent_prompt is None:
        return f"Agent {agent_name} not found."
    model = InstructModel()
    combined_input = f"{agent_prompt}\n\n:User  {input_text}\nAgent:"
    return model.generate_response(combined_input)

def workspace_interface(project_name):
    project_path = os.path.join(PROJECT_ROOT, project_name)
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
        st.session_state.workspace_projects[project_name] = {"files": []}
        st.session_state.current_state['workspace_chat']['project_name'] = project_name
        commit_and_push_changes(f"Create project {project_name}")
        return f"Project {project_name} created successfully."
    return f"Project {project_name} already exists."

def add_code_to_workspace(project_name, code, file_name):
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if os.path.exists(project_path):
        file_path = os.path.join(project_path, file_name)
        with open(file_path, "w") as file:
            file.write(code)
        st.session_state.workspace_projects[project_name]["files"].append(file_name)
        st.session_state.current_state['workspace_chat']['added_code'] = {"file_name": file_name, "code": code}
        commit_and_push_changes(f"Add code to {file_name} in project {project_name}")
        return f"Code added to {file_name} in project {project_name} successfully."
    return f"Project {project_name} does not exist."

def terminal_interface(command, project_name=None):
    if project_name:
        project_path = os.path.join(PROJECT_ROOT, project_name)
        if not os.path.exists(project_path):
            return f"Project {project_name} does not exist."
        result = subprocess.run(command, cwd=project_path, shell=True, capture_output=True, text=True)
    else:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout if result.returncode == 0 else result.stderr
    st.session_state.current_state['toolbox']['terminal_output'] = output
    return output

def code_editor_interface(code):
    formatted_code = format_code(code)
    lint_message = lint_code(formatted_code)
    return formatted_code, lint_message

def format_code(code):
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.NothingChanged:
        return code
    except Exception as e:
        return None, f"Error formatting code with black: {e}"

def lint_code(code):
    result = StringIO()
    sys.stdout = result
    sys.stderr = result
    try:
        (pylint_stdout, pylint_stderr) = lint.py_run(code, return_std=True)
        return pylint_stdout.getvalue() + pylint_stderr.getvalue()
    except Exception as e:
        return None, f"Error linting code with pylint: {e}"
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def translate_code(code, input_language, output_language):
    model = InstructModel()
    prompt = f"Translate the following {input_language} code to {output_language}:\n\n{code}"
    return model.generate_response(prompt)

def generate_code(code_idea):
    model = InstructModel()
    prompt = f"Generate code for the following idea:\n\n{code_idea}"
    generated_code = model.generate_response(prompt)
    st.session_state.current_state['toolbox']['generated_code'] = generated_code
    return generated_code

def commit_and_push_changes(commit_message):
    """Commits and pushes changes to the Hugging Face repository."""
    try:
        subprocess.run(["git", "add", "."], check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True, capture_output=True, text=True)
        subprocess.run(["git", "push"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Git command failed: {e.stderr}")
    except FileNotFoundError:
        st.error("Git not found. Please ensure Git is installed and configured.")

# Streamlit App
st.title("AI Agent Creator")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["AI Agent Creator", "Tool Box", "Workspace Chat App"])

if app_mode == "AI Agent Creator":
    st.header("Create an AI Agent from Text")
    agent_name = st.text_input("Enter agent name:")
    text_input = st.text_area("Enter skills (one per line):")
    if st.button("Create Agent"):
        agent_prompt = create_agent_from_text(agent_name, text_input)
        st.success(f"Agent '{agent_name}' created and saved successfully.")
        st.session_state.available_agents.append(agent_name)

elif app_mode == "Tool Box":
    st.header("AI-Powered Tools")

    # Chat Interface
    st.subheader("Chat with CodeCraft")
    chat_input = st.text_area("Enter your message:")
    if st.button("Send"):
        if chat_input.startswith("@"):
            agent_name = chat_input.split(" ")[0][1:]  # Extract agent_name from @agent_name
            chat_input = " ".join(chat_input.split(" ")[1:])  # Remove agent_name from input
            chat_response = chat_interface_with_agent(chat_input, agent_name)
        else:
            chat_response = chat_interface(chat_input)
        st.session_state.chat_history.append((chat_input, chat_response))
        st.write(f"CodeCraft: {chat_response}")

    # Terminal Interface
    st.subheader("Terminal")
    terminal_input = st.text_input("Enter a command:")
    if st.button("Run"):
        terminal_output = terminal_interface(terminal_input)
        st.session_state.terminal_history.append((terminal_input, terminal_output))
        st.code(terminal_output, language="bash")

    # Code Editor Interface
    st.subheader("Code Editor")
    code_editor = st.text_area("Write your code:", height=300)
    if st.button("Format & Lint"):
        formatted_code, lint_message = code_editor_interface(code_editor)
        st.code(formatted_code, language="python")
        st.info(lint_message)

    # Text Translation Tool (Code Translation)
    st.subheader("Translate Code")
    code_to_translate = st.text_area("Enter code to translate:")
    source_language = st.text_input("Enter source language (e.g., 'Python'):")
    target_language = st.text_input("Enter target language (e.g., 'JavaScript'):")
    if st.button("Translate Code"):
        translated_code = translate_code(code_to_translate, source_language, target_language)
        st.code(translated_code, language=target_language.lower())

    # Code Generation
    st.subheader("Code Generation")
    code_idea = st.text_input("Enter your code idea:")
    if st.button("Generate Code"):
        generated_code = generate_code(code_idea)
        st.code(generated_code, language="python")

elif app_mode == "Workspace Chat App":
    st.header("Workspace Chat App")

    # Project Workspace Creation
    st.subheader("Create a New Project")
    project_name = st.text_input("Enter project name:")
    if st.button("Create Project"):
        workspace_status = workspace_interface(project_name)
        st.success(workspace_status)

    # Add Code to Workspace
    st.subheader("Add Code to Workspace")
    code_to_add = st.text_area("Enter code to add to workspace:")
    file_name = st.text_input("Enter file name (e.g., 'app.py'):")
    if st.button("Add Code"):
        add_code_status = add_code_to_workspace(project_name, code_to_add, file_name)
        st.success(add_code_status)

    # Terminal Interface with Project Context
    st.subheader("Terminal (Workspace Context)")
    terminal_input = st.text_input("Enter a command within the workspace:")
    if st.button("Run Command"):
        terminal_output = terminal_interface(terminal_input, project_name)
        st.code(terminal_output, language="bash")

    # Chat Interface for Guidance
    st.subheader("Chat with CodeCraft for Guidance")
    chat_input = st.text_area("Enter your message for guidance:")
    if st.button("Get Guidance"):
        chat_response = chat_interface(chat_input)
        st.session_state.chat_history.append((chat_input, chat_response))
        st.write(f"CodeCraft: {chat_response}")

    # Display Chat History
    st.subheader("Chat History")
    for user_input, response in st.session_state.chat_history:
        st.write(f":User  {user_input}")
        st.write(f"CodeCraft: {response}")

    # Display Terminal History
    st.subheader("Terminal History")
    for command, output in st.session_state.terminal_history:
        st.write(f"Command: {command}")
        st.code(output, language="bash")

    # Display Projects and Files
    st.subheader("Workspace Projects")
    for project, details in st.session_state.workspace_projects.items():
        st.write(f"Project: {project }")
        for file in details['files']:
            st.write(f"  - {file}")

    # Chat with AI Agents
    st.subheader("Chat with AI Agents")
    selected_agent = st.selectbox("Select an AI agent", st.session_state.available_agents)
    agent_chat_input = st.text_area("Enter your message for the agent:")
    if st.button("Send to Agent"):
        agent_chat_response = chat_interface_with_agent(agent_chat_input, selected_agent)
        st.session_state.chat_history.append((agent_chat_input, agent_chat_response))
        st.write(f"{selected_agent}: {agent_chat_response}")

    # Automate Build Process
    st.subheader("Automate Build Process")
    if st.button("Automate"):
        agent = AIAgent(selected_agent, "", [])  # Load the agent without skills for now
        summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects)
        st.write("Autonomous Build Summary:")
        st.write(summary)
        st.write("Next Step:")
        st.write(next_step)