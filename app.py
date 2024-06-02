import os
import subprocess
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import black
from pylint import lint
from io import StringIO
import openai
import sys

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

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

class AIAgent:
    def __init__(self, name, description, skills):
        self.name = name
        self.description = description
        self.skills = skills

    def create_agent_prompt(self):
        skills_str = '\n'.join([f"* {skill}" for skill in self.skills])
        agent_prompt = f"""
As an elite expert developer, my name is {self.name}. I possess a comprehensive understanding of the following areas:
{skills_str}

I am confident that I can leverage my expertise to assist you in developing and deploying cutting-edge web applications. Please feel free to ask any questions or present any challenges you may encounter.
"""
        return agent_prompt

    def autonomous_build(self, chat_history, workspace_projects):
        """
        Autonomous build logic that continues based on the state of chat history and workspace projects.
        """
        summary = "Chat History:\n" + "\n".join([f"User: {u}\nAgent: {a}" for u, a in chat_history])
        summary += "\n\nWorkspace Projects:\n" + "\n".join([f"{p}: {details}" for p, details in workspace_projects.items()])

        # Analyze chat history and workspace projects to suggest actions
        # Example:
        # - Check if the user has requested to create a new file
        # - Check if the user has requested to install a package
        # - Check if the user has requested to run a command
        # - Check if the user has requested to generate code
        # - Check if the user has requested to translate code
        # - Check if the user has requested to summarize text
        # - Check if the user has requested to analyze sentiment

        # Generate a response based on the analysis
        next_step = "Based on the current state, the next logical step is to implement the main application logic."

        return summary, next_step

def save_agent_to_file(agent):
    """Saves the agent's prompt to a file."""
    if not os.path.exists(AGENT_DIRECTORY):
        os.makedirs(AGENT_DIRECTORY)
    file_path = os.path.join(AGENT_DIRECTORY, f"{agent.name}.txt")
    with open(file_path, "w") as file:
        file.write(agent.create_agent_prompt())
    st.session_state.available_agents.append(agent.name)

def load_agent_prompt(agent_name):
    """Loads an agent prompt from a file."""
    file_path = os.path.join(AGENT_DIRECTORY, f"{agent_name}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            agent_prompt = file.read()
        return agent_prompt
    else:
        return None

def create_agent_from_text(name, text):
    skills = text.split('\n')
    agent = AIAgent(name, "AI agent created from text input.", skills)
    save_agent_to_file(agent)
    return agent.create_agent_prompt()

def chat_interface_with_agent(input_text, agent_name):
    agent_prompt = load_agent_prompt(agent_name)
    if agent_prompt is None:
        return f"Agent {agent_name} not found."

    model_name ="MaziyarPanahi/Codestral-22B-v0.1-GGUF"
    try:
        model = AutoModel.from_pretrained("MaziyarPanahi/Codestral-22B-v0.1-GGUF")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    combined_input = f"{agent_prompt}\n\nUser: {input_text}\nAgent:"
    
    input_ids = tokenizer.encode(combined_input, return_tensors="pt")
    max_input_length = 900
    if input_ids.shape[1] > max_input_length:
        input_ids = input_ids[:, :max_input_length]

    outputs = model.generate(
        input_ids, max_new_tokens=50, num_return_sequences=1, do_sample=True,
        pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Terminal interface
def terminal_interface(command, project_name=None):
    if project_name:
        project_path = os.path.join(PROJECT_ROOT, project_name)
        if not os.path.exists(project_path):
            return f"Project {project_name} does not exist."
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=project_path)
    else:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

# Code editor interface
def code_editor_interface(code):
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
    except black.NothingChanged:
        formatted_code = code

    result = StringIO()
    sys.stdout = result
    sys.stderr = result

    (pylint_stdout, pylint_stderr) = lint.py_run(code, return_std=True)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    lint_message = pylint_stdout.getvalue() + pylint_stderr.getvalue()

    return formatted_code, lint_message

# Text summarization tool
def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Sentiment analysis tool
def sentiment_analysis(text):
    analyzer = pipeline("sentiment-analysis")
    result = analyzer(text)
    return result[0]['label']

# Text translation tool (code translation)
def translate_code(code, source_language, target_language):
    # Use a Hugging Face translation model instead of OpenAI
    translator = pipeline("translation", model="bartowski/Codestral-22B-v0.1-GGUF")  # Example: English to Spanish
    translated_code = translator(code, target_lang=target_language)[0]['translation_text']
    return translated_code

def generate_code(code_idea):
    # Use a Hugging Face code generation model instead of OpenAI
    generator = pipeline('text-generation', model='bigcode/starcoder')
    generated_code = generator(code_idea, max_length=1000, num_return_sequences=1)[0]['generated_text']
    return generated_code

def chat_interface(input_text):
    """Handles general chat interactions with the user."""
    # Use a Hugging Face chatbot model or your own logic
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    response = chatbot(input_text, max_length=50, num_return_sequences=1)[0]['generated_text']
    return response

# Workspace interface
def workspace_interface(project_name):
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
        st.session_state.workspace_projects[project_name] = {'files': []}
        return f"Project '{project_name}' created successfully."
    else:
        return f"Project '{project_name}' already exists."

# Add code to workspace
def add_code_to_workspace(project_name, code, file_name):
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(project_path):
        return f"Project '{project_name}' does not exist."
    
    file_path = os.path.join(project_path, file_name)
    with open(file_path, "w") as file:
        file.write(code)
    st.session_state.workspace_projects[project_name]['files'].append(file_name)
    return f"Code added to '{file_name}' in project '{project_name}'."

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
        agent_prompt = create_agent_from_text(agent_name, text_input)
        st.success(f"Agent '{agent_name}' created and saved successfully.")
        st.session_state.available_agents.append(agent_name)

elif app_mode == "Tool Box":
    # Tool Box
    st.header("AI-Powered Tools")

    # Chat Interface
    st.subheader("Chat with CodeCraft")
    chat_input = st.text_area("Enter your message:")
    if st.button("Send"):
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

    # Text Summarization Tool
    st.subheader("Summarize Text")
    text_to_summarize = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        summary = summarize_text(text_to_summarize)
        st.write(f"Summary: {summary}")

    # Sentiment Analysis Tool
    st.subheader("Sentiment Analysis")
    sentiment_text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        sentiment = sentiment_analysis(sentiment_text)
        st.write(f"Sentiment: {sentiment}")

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
    # Workspace Chat App
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
        st.write(f"User: {user_input}")
        st.write(f"CodeCraft: {response}")

    # Display Terminal History
    st.subheader("Terminal History")
    for command, output in st.session_state.terminal_history:
        st.write(f"Command: {command}")
        st.code(output, language="bash")

    # Display Projects and Files
    st.subheader("Workspace Projects")
    for project, details in st.session_state.workspace_projects.items():
        st.write(f"Project: {project}")
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