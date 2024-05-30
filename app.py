import os
import sys
import subprocess
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import black
from pylint import lint
from io import StringIO
import openai

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

        next_step = "Based on the current state, the next logical step is to implement the main application logic."

        return summary, next_step

def save_agent_to_file(agent):
    """Saves the agent's prompt to a file locally and then commits to the Hugging Face repository."""
    if not os.path.exists(AGENT_DIRECTORY):
        os.makedirs(AGENT_DIRECTORY)
    file_path = os.path.join(AGENT_DIRECTORY, f"{agent.name}.txt")
    config_path = os.path.join(AGENT_DIRECTORY, f"{agent.name}Config.txt")
    with open(file_path, "w") as file:
        file.write(agent.create_agent_prompt())
    with open(config_path, "w") as file:
        file.write(f"Agent Name: {agent.name}\nDescription: {agent.description}")
    st.session_state.available_agents.append(agent.name)

    commit_and_push_changes(f"Add agent {agent.name}")

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
        input_ids, max_new_tokens=50, num_return_sequences=1, do_sample=True, pad_token_id=tokenizer.eos_token_id # Set pad_token_id to eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def workspace_interface(project_name):
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(PROJECT_ROOT):
        os.makedirs(PROJECT_ROOT)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
        st.session_state.workspace_projects[project_name] = {"files": []}
        st.session_state.current_state['workspace_chat']['project_name'] = project_name
        commit_and_push_changes(f"Create project {project_name}")
        return f"Project {project_name} created successfully."
    else:
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
    else:
        return f"Project {project_name} does not exist."

def terminal_interface(command, project_name=None):
    if project_name:
        project_path = os.path.join(PROJECT_ROOT, project_name)
        if not os.path.exists(project_path):
            return f"Project {project_name} does not exist."
        result = subprocess.run(command, cwd=project_path, shell=True, capture_output=True, text=True)
    else:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        st.session_state.current_state['toolbox']['terminal_output'] = result.stdout
        return result.stdout
    else:
        st.session_state.current_state['toolbox']['terminal_output'] = result.stderr
        return result.stderr

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
    st.session_state.current_state['toolbox']['formatted_code'] = formatted_code
    st.session_state.current_state['toolbox']['lint_message'] = lint_message
    return formatted_code, lint_message

def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    st.session_state.current_state['toolbox']['summary'] = summary[0]['summary_text']
    return summary[0]['summary_text']

def sentiment_analysis(text):
    analyzer = pipeline("sentiment-analysis")
    sentiment = analyzer(text)
    st.session_state.current_state['toolbox']['sentiment'] = sentiment[0]
    return sentiment[0]

def translate_code(code, input_language, output_language):
    # Define a dictionary to map programming languages to their corresponding file extensions
    language_extensions = {
        # ignore the specific languages right now, and continue to EOF
    }

    # Add code to handle edge cases such as invalid input and unsupported programming languages
    if input_language not in language_extensions:
        raise ValueError(f"Invalid input language: {input_language}")
    if output_language not in language_extensions:
        raise ValueError(f"Invalid output language: {output_language}")

    # Use the dictionary to map the input and output languages to their corresponding file extensions
    input_extension = language_extensions[input_language]
    output_extension = language_extensions[output_language]

    # Translate the code using the OpenAI API
    prompt = f"Translate this code from {input_language} to {output_language}:\n\n{code}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert software developer."},
            {"role": "user", "content": prompt}
        ]
    )
    translated_code = response.choices[0].message['content'].strip()

    # Return the translated code
    translated_code = response.choices[0].message['content'].strip()
    st.session_state.current_state['toolbox']['translated_code'] = translated_code
    return translated_code

def generate_code(code_idea):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert software developer."},
            {"role": "user", "content": f"Generate a Python code snippet for the following idea:\n\n{code_idea}"}
        ]
    )
    generated_code = response.choices[0].message['content'].strip()
    st.session_state.current_state['toolbox']['generated_code'] = generated_code
    return generated_code

def commit_and_push_changes(commit_message):
    """Commits and pushes changes to the Hugging Face repository."""
    commands = [
        "git add .",
        f"git commit -m '{commit_message}'",
        "git push"
    ]
    for command in commands:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Error executing command '{command}': {result.stderr}")
            break

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
    preset_commands = {
        "Create a new project": "create_project('project_name')",
        "Add code to workspace": "add_code_to_workspace('project_name', 'code', 'file_name')",
        "Run terminal command": "terminal_interface('command', 'project_name')",
        "Generate code": "generate_code('code_idea')",
        "Summarize text": "summarize_text('text')",
        "Analyze sentiment": "sentiment_analysis('text')",
        "Translate code": "translate_code('code', 'source_language', 'target_language')",
    }
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

# Display current state for debugging
st.sidebar.subheader("Current State")
st.sidebar.json(st.session_state.current_state)