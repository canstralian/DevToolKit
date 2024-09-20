import subprocess
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import black
from pylint import lint
from io import StringIO

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
        Autonomous build logic. 
        For now, it provides a simple summary and suggests the next step.
        """
        summary = "Chat History:\n" + "\n".join([f"User: {u}\nAgent: {a}" for u, a in chat_history])
        summary += "\n\nWorkspace Projects:\n" + "\n".join(
            [f"{p}: {details}" for p, details in workspace_projects.items()])

        next_step = "Based on the current state, the next logical step is to implement the main application logic."

        return summary, next_step


def save_agent_to_file(agent):
    """Saves the agent's information to files."""
    if not os.path.exists(AGENT_DIRECTORY):
        os.makedirs(AGENT_DIRECTORY)
    file_path = os.path.join(AGENT_DIRECTORY, f"{agent.name}.txt")
    config_path = os.path.join(AGENT_DIRECTORY, f"{agent.name}Config.txt")
    with open(file_path, "w") as file:
        file.write(agent.create_agent_prompt())
    with open(config_path, "w") as file:
        file.write(f"Agent Name: {agent.name}\nDescription: {agent.description}")
    st.session_state.available_agents.append(agent.name)

    # (Optional) Commit and push if you have set up Hugging Face integration.
    # commit_and_push_changes(f"Add agent {agent.name}")


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
    """Creates an AI agent from the provided text input."""
    skills = text.split('\n')
    agent = AIAgent(name, "AI agent created from text input.", skills)
    save_agent_to_file(agent)
    return agent.create_agent_prompt()


# Chat interface using a selected agent
def chat_interface_with_agent(input_text, agent_name):
    agent_prompt = load_agent_prompt(agent_name)
    if agent_prompt is None:
        return f"Agent {agent_name} not found."

    # Load the GPT-2 model 
    model_name = "gpt2"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Combine agent prompt and user input (truncate if necessary)
    combined_input = f"{agent_prompt}\n\nUser: {input_text}\nAgent:"
    max_input_length = 900 
    input_ids = tokenizer.encode(combined_input, return_tensors="pt")
    if input_ids.shape[1] > max_input_length:
        input_ids = input_ids[:, :max_input_length]

    # Generate response
    outputs = model.generate(
        input_ids, 
        max_new_tokens=50, 
        num_return_sequences=1, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id 
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Basic chat interface (no agent)
def chat_interface(input_text):
    # Load the GPT-2 model 
    model_name = "gpt2"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Generate response
    outputs = generator(input_text, max_new_tokens=50, num_return_sequences=1, do_sample=True)
    response = outputs[0]['generated_text']
    return response


def workspace_interface(project_name):
    """Manages project creation."""
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(PROJECT_ROOT):
        os.makedirs(PROJECT_ROOT)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
        st.session_state.workspace_projects[project_name] = {"files": []}
        st.session_state.current_state['workspace_chat']['project_name'] = project_name
        # (Optional) Commit and push if you have set up Hugging Face integration.
        # commit_and_push_changes(f"Create project {project_name}")
        return f"Project {project_name} created successfully."
    else:
        return f"Project {project_name} already exists."


def add_code_to_workspace(project_name, code, file_name):
    """Adds code to a file in the specified project."""
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if os.path.exists(project_path):
        file_path = os.path.join(project_path, file_name)
        with open(file_path, "w") as file:
            file.write(code)
        st.session_state.workspace_projects[project_name]["files"].append(file_name)
        st.session_state.current_state['workspace_chat']['added_code'] = {"file_name": file_name, "code": code}
        # (Optional) Commit and push if you have set up Hugging Face integration.
        # commit_and_push_changes(f"Add code to {file_name} in project {project_name}")
        return f"Code added to {file_name} in project {project_name} successfully."
    else:
        return f"Project {project_name} does not exist."

def terminal_interface(command, project_name=None):
    """Executes commands in the terminal, optionally within a project's directory."""
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


def summarize_text(text):
    """Summarizes text using a Hugging Face pipeline."""
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=100, min_length=25, do_sample=False)
    st.session_state.current_state['toolbox']['summary'] = summary[0]['summary_text']
    return summary[0]['summary_text']


def sentiment_analysis(text):
    """Analyzes sentiment of text using a Hugging Face pipeline."""
    analyzer = pipeline("sentiment-analysis")
    sentiment = analyzer(text)
    st.session_state.current_state['toolbox']['sentiment'] = sentiment[0]
    return sentiment[0]


def code_editor_interface(code):
    """Formats and lints Python code."""
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
        lint_result = StringIO()
        lint.Run([
            '--disable=C0114,C0115,C0116', 
            '--output-format=text',
            '--reports=n',
            '-' 
        ], exit=False, do_exit=False)
        lint_message = lint_result.getvalue()
        return formatted_code, lint_message
    except Exception as e:
        return code, f"Error formatting or linting code: {e}"


def translate_code(code, input_language, output_language):
    """Translates code between programming languages."""
    try:
        translator = pipeline("translation", model=f"{input_language}-to-{output_language}")
        translated_code = translator(code, max_length=10000)[0]['translation_text']
        st.session_state.current_state['toolbox']['translated_code'] = translated_code
        return translated_code
    except Exception as e:
        return f"Error translating code: {e}"


def generate_code(code_idea):
    """Generates code from a user idea using a Hugging Face pipeline."""
    try:
        generator = pipeline('text-generation', model='gpt2') 
        generated_code = generator(f"```python\n{code_idea}\n```", max_length=1000, num_return_sequences=1)[0][
            'generated_text']

        # Extract code from the generated text
        start_index = generated_code.find("```python") + len("```python")
        end_index = generated_code.find("```", start_index)
        if start_index != -1 and end_index != -1:
            generated_code = generated_code[start_index:end_index].strip()

        st.session_state.current_state['toolbox']['generated_code'] = generated_code
        return generated_code
    except Exception as e:
        return f"Error generating code: {e}"


def commit_and_push_changes(commit_message):
    """(Optional) Commits and pushes changes. 
    Needs to be configured for your Hugging Face repository.
    """
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


# --- Streamlit App ---
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

    st.subheader("Chat with CodeCraft")
    chat_input = st.text_area("Enter your message:")
    if st.button("Send"):
        if chat_input.startswith("@"):
            agent_name = chat_input.split(" ")[0][1:] 
            chat_input = " ".join(chat_input.split(" ")[1:])  
            chat_response = chat_interface_with_agent(chat_input, agent_name)
        else:
            chat_response = chat_interface(chat_input)
        st.session_state.chat_history.append((chat_input, chat_response))
        st.write(f"CodeCraft: {chat_response}")

    st.subheader("Terminal")
    terminal_input = st.text_input("Enter a command:")
    if st.button("Run"):
        terminal_output = terminal_interface(terminal_input)
        st.session_state.terminal_history.append((terminal_input, terminal_output))
        st.code(terminal_output, language="bash")

    st.subheader("Code Editor")
    code_editor = st.text_area("Write your code:", height=300)
    if st.button("Format & Lint"):
        formatted_code, lint_message = code_editor_interface(code_editor)
        st.code(formatted_code, language="python")
        st.info(lint_message)

    st.subheader("Summarize Text")
    text_to_summarize = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        summary = summarize_text(text_to_summarize)
        st.write(f"Summary: {summary}")

    st.subheader("Sentiment Analysis")
    sentiment_text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        sentiment = sentiment_analysis(sentiment_text)
        st.write(f"Sentiment: {sentiment}")

    st.subheader("Translate Code")
    code_to_translate = st.text_area("Enter code to translate:")
    source_language = st.selectbox("Source Language", ["en", "fr", "de", "es", "zh", "ja", "ko", "ru"])
    target_language = st.selectbox("Target Language", ["en", "fr", "de", "es", "zh", "ja", "ko", "ru"])
    if st.button("Translate Code"):
        translated_code = translate_code(code_to_translate, source_language, target_language)
        st.code(translated_code, language=target_language.lower())

    st.subheader("Code Generation")
    code_idea = st.text_input("Enter your code idea:")
    if st.button("Generate Code"):
        generated_code = generate_code(code_idea)
        st.code(generated_code, language="python")

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
    st.header("Workspace Chat App")

    st.subheader("Create a New Project")
    project_name = st.text_input("Enter project name:")
    if st.button("Create Project"):
        workspace_status = workspace_interface(project_name)
        st.success(workspace_status)

    st.subheader("Add Code to Workspace")
    code_to_add = st.text_area("Enter code to add to workspace:")
    file_name = st.text_input("Enter file name (e.g. 'app.py'):")
    if st.button("Add Code"):
        add_code_status = add_code_to_workspace(project_name, code_to_add, file_name)
        st.success(add_code_status)

    st.subheader("Terminal (Workspace Context)")
    terminal_input = st.text_input("Enter a command within the workspace:")
    if st.button("Run Command"):
        terminal_output = terminal_interface(terminal_input, project_name)
        st.code(terminal_output, language="bash")

    st.subheader("Chat with CodeCraft for Guidance")
    chat_input = st.text_area("Enter your message for guidance:")
    if st.button("Get Guidance"):
        chat_response = chat_interface(chat_input)
        st.session_state.chat_history.append((chat_input, chat_response))
        st.write(f"CodeCraft: {chat_response}")

    st.subheader("Chat History")
    for user_input, response in st.session_state.chat_history:
        st.write(f"User: {user_input}")
        st.write(f"CodeCraft: {response}")

    st.subheader("Terminal History")
    for command, output in st.session_state.terminal_history:
        st.write(f"Command: {command}")
        st.code(output, language="bash")

    st.subheader("Workspace Projects")
    for project, details in st.session_state.workspace_projects.items():
        st.write(f"Project: {project}")
        for file in details['files']:
            st.write(f"  - {file}")

    st.subheader("Chat with AI Agents")
    selected_agent = st.selectbox("Select an AI agent", st.session_state.available_agents)
    agent_chat_input = st.text_area("Enter your message for the agent:")
    if st.button("Send to Agent"):
        agent_chat_response = chat_interface_with_agent(agent_chat_input, selected_agent)
        st.session_state.chat_history.append((agent_chat_input, agent_chat_response))
        st.write(f"{selected_agent}: {agent_chat_response}")

    st.subheader("Automate Build Process")
    if st.button("Automate"):
        if selected_agent:
            agent = AIAgent(selected_agent, "", []) 
            summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects)
            st.write("Autonomous Build Summary:")
            st.write(summary)
            st.write("Next Step:")
            st.write(next_step)
        else:
            st.warning("Please select an AI agent first.")
