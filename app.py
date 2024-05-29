import streamlit as st
import os
import subprocess
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import black
from pylint import lint
from io import StringIO
import openai
import sys

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

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

# Chat interface using a selected agent
def chat_interface_with_agent(input_text, agent_name):
    agent_prompt = load_agent_prompt(agent_name)
    if agent_prompt is None:
        return f"Agent {agent_name} not found."

    # Load the GPT-2 model which is compatible with AutoModelForCausalLM
    model_name = "gpt2"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Combine the agent prompt with user input
    combined_input = f"{agent_prompt}\n\nUser: {input_text}\nAgent:"
    
    # Truncate input text to avoid exceeding the model's maximum length
    max_input_length = 900
    input_ids = tokenizer.encode(combined_input, return_tensors="pt")
    if input_ids.shape[1] > max_input_length:
        input_ids = input_ids[:, :max_input_length]

    # Generate chatbot response
    outputs = model.generate(
        input_ids, max_new_tokens=50, num_return_sequences=1, do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Preset commands for no-code-knowledge developers
preset_commands = {
    "Create a new project": "create_project('project_name')",
    "Add code to workspace": "add_code_to_workspace('project_name', 'code', 'file_name')",
    "Run terminal command": "terminal_interface('command', 'project_name')",
    "Generate code": "generate_code('code_idea')",
    "Summarize text": "summarize_text('text')",
    "Analyze sentiment": "sentiment_analysis('text')",
    "Translate code": "translate_code('code', 'source_language', 'target_language')",
}

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
    source_language = st.text_input("Enter source language (e.g. 'Python'):")
    target_language = st.text_input("Enter target language (e.g. 'JavaScript'):")
    if st.button("Translate Code"):
        translated_code = translate_code(code_to_translate, source_language, target_language)
        st.code(translated_code, language=target_language.lower())

    # Code Generation
    st.subheader("Code Generation")
    code_idea = st.text_input("Enter your code idea:")
    if st.button("Generate Code"):
        generated_code = generate_code(code_idea)
        st.code(generated_code, language="python")

    # Display Preset Commands
    st.subheader("Preset Commands")
    for command_name, command in preset_commands.items():
        st.write(f"{command_name}: `{command}`")

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
    file_name = st.text_input("Enter file name (e.g. 'app.py'):")
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