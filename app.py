import os
import streamlit as st
import subprocess
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel, RagRetriever, AutoModelForSeq2SeqLM
import black
from pylint import lint
from io import StringIO
import sys
import torch
from huggingface_hub import hf_hub_url, cached_download, HfApi
import base64

# Add the new HTML code below
custom_html = '''
<div style='position:fixed;bottom:0;left:0;width:100%;'>
  <iframe width="100%" scrolling="no" title="CodeGPT Widget" frameborder="0" allowtransparency sandbox="" allowfullscreen="" data-widget-id="c265505c-e667-4af2-b492-291da888ee7c" src="https://widget.codegpt.co/chat-widget.js"></iframe>
</div>'''

# Update the markdown function to accept custom HTML code
def markdown_with_custom_html(md, html):
    md_content = md
    if html:
        return f"{md_content}\n\n{html}"
    else:
        return md_content


markdown_text = "Compare model responses with me!"
markdown_with_custom_html(markdown_text, custom_html)


# Set your Hugging Face API key here
# hf_token = "YOUR_HUGGING_FACE_API_KEY"  # Replace with your actual token
# Get Hugging Face token from secrets.toml - this line should already be in the main code
hf_token = st.secrets["huggingface"]["hf_token"]

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
rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-base")  # Use a Hugging Face RAG model

# Load pre-trained chat model
chat_model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/DialoGPT-medium")  # Use a Hugging Face chat model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

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

        # Ensure project folder exists
        project_path = os.path.join(PROJECT_ROOT, project_name)
        if not os.path.exists(project_path):
            os.makedirs(project_path)

        # Create requirements.txt if it doesn't exist
        requirements_file = os.path.join(project_path, "requirements.txt")
        if not os.path.exists(requirements_file):
            with open(requirements_file, "w") as f:
                f.write("# Add your project's dependencies here\n")

        # Create app.py if it doesn't exist
        app_file = os.path.join(project_path, "app.py")
        if not os.path.exists(app_file):
            with open(app_file, "w") as f:
                f.write("# Your project's main application logic goes here\n")

        # Generate GUI code for app.py if requested
        if "create a gui" in summary.lower():
            gui_code = generate_code("Create a simple GUI for this application", selected_model)
            with open(app_file, "a") as f:
                f.write(gui_code)

        # Run the default build process
        build_command = "pip install -r requirements.txt && python app.py"
        try:
            result = subprocess.run(build_command, shell=True, capture_output=True, text=True, cwd=project_path)
            st.write(f"Build Output:\n{result.stdout}")
            if result.stderr:
                st.error(f"Build Errors:\n{result.stderr}")
        except Exception as e:
            st.error(f"Build Error: {e}")

        return summary, next_step
    
    def deploy_built_space_to_hf(self):
        if not self._hf_api or not self._hf_token:
            raise ValueError("Cannot deploy the Space since no valid Hugoging Face API connection was established.")

        # Assuming you have a function to get the files for your Space
        repository_name = f"my-awesome-space_{datetime.now().timestamp()}" 
        files = get_built_space_files() # Placeholder - you'll need to define this function

        # Create the Space
        create_space(self.hf_api, repository_name, "Description", True, files) 

        st.markdown("## Congratulations! Successfully deployed Space 🚀 ##")
        st.markdown(f"[Check out your new Space here](https://huggingface.co/spaces/{repository_name})")


# Add any missing functions from your original code (e.g., get_built_space_files)
def get_built_space_files():
    # Replace with your logic to gather the files you want to deploy
    return {
        "app.py": "# Your Streamlit app code here",
        "requirements.txt": "streamlit\ntransformers" 
        # Add other files as needed
    }

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
        from transformers import AutoModel, AutoTokenizer  # Import AutoModel here
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

def generate_code(code_idea, model_name):
    """Generates code using the selected model."""
    try:
        generator = pipeline('text-generation', model=model_name)
        generated_code = generator(code_idea, max_length=1000, num_return_sequences=1)[0]['generated_text']
        return generated_code
    except Exception as e:
        return f"Error generating code: {e}"

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

def create_space(api, name, description, public, files, entrypoint="launch.py"):
    url = f"{hf_hub_url()}spaces/{name}/prepare-repo"
    headers = {"Authorization": f"Bearer {api.access_token}"}
    payload = {
        "public": public,
        "gitignore_template": "web",
        "default_branch": "main",
        "archived": False,
        "files": []
    }
    for filename, contents in files.items():
        data = {
            "content": contents,
            "path": filename,
            "encoding": "utf-8",
            "mode": "overwrite" if "#\{random.randint(0, 1)\}" not in contents else "merge",
        }
        payload["files"].append(data)
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    location = response.headers.get("Location")
    # wait_for_processing(location, api)  # You might need to implement this if it's not already defined

    return Repository(name=name, api=api)

# Streamlit App
st.title("AI Agent Creator")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["AI Agent Creator", "Tool Box", "Workspace Chat App"])

# Get Hugging Face token from secrets.toml
hf_token = st.secrets["huggingface"]["hf_token"]

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
def get_built_space_files():
    """
    Gathers the necessary files for the Hugging Face Space, 
    handling different project structures and file types.
    """
    files = {}

    # Get the current project name (adjust as needed)
    project_name = st.session_state.get('project_name', 'my_project') 
    project_path = os.path.join(PROJECT_ROOT, project_name)

    # Define a list of files/directories to search for
    targets = [
        "app.py", 
        "requirements.txt",
        "Dockerfile",  
        "docker-compose.yml", # Example YAML file
        "src",          # Example subdirectory
        "assets"        # Another example subdirectory
    ]

    # Iterate through the targets
    for target in targets:
        target_path = os.path.join(project_path, target)

        # If the target is a file, add it to the files dictionary
        if os.path.isfile(target_path):
            add_file_to_dictionary(files, target_path)

        # If the target is a directory, recursively search for files within it
        elif os.path.isdir(target_path):
            for root, _, filenames in os.walk(target_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    add_file_to_dictionary(files, file_path)

    return files

def add_file_to_dictionary(files, file_path):
    """Helper function to add a file to the files dictionary."""
    filename = os.path.relpath(file_path, PROJECT_ROOT) # Get relative path

    # Handle text and binary files
    if filename.endswith((".py", ".txt", ".json", ".html", ".css", ".yml", ".yaml")):
        with open(file_path, "r") as f:
            files[filename] = f.read()
    else:
        with open(file_path, "rb") as f:
            file_content = f.read()
            files[filename] = base64.b64encode(file_content).decode("utf-8")
    # Project Workspace Creation
    st.subheader("Create a New Project")
    project_name = st.text_input("Enter project name:")
    if st.button("Create Project"):
        workspace_status = workspace_interface(project_name)
        st.success(workspace_status)

        # Automatically create requirements.txt and app.py
        project_path = os.path.join(PROJECT_ROOT, project_name)
        requirements_file = os.path.join(project_path, "requirements.txt")
        if not os.path.exists(requirements_file):
            with open(requirements_file, "w") as f:
                f.write("# Add your project's dependencies here\n")

        app_file = os.path.join(project_path, "app.py")
        if not os.path.exists(app_file):
            with open(app_file, "w") as f:
                f.write("# Your project's main application logic goes here\n")

    # Add Code to Workspace
    st.subheader("Add Code to Workspace")
    code_to_add = st.text_area("Enter code to add to workspace:")
    file_name = st.text_input("Enter file name (e.g., 'app.py'):")
    if st.button("Add Code"):
        add_code_status = add_code_to_workspace(project_name, code_to_add, file_name)
        st.session_state.terminal_history.append((f"Add Code: {code_to_add}", add_code_status))
        st.success(add_code_status)

    # Terminal Interface with Project Context
    st.subheader("Terminal (Workspace Context)")
    terminal_input = st.text_input("Enter a command within the workspace:")
    if st.button("Run Command"):
        terminal_output = terminal_interface(terminal_input, project_name)
        st.session_state.terminal_history.append((terminal_input, terminal_output))
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

    # Code Generation
    st.subheader("Code Generation")
    code_idea = st.text_input("Enter your code idea:")

    # Model Selection Menu
    selected_model = st.selectbox("Select a code-generative model", AVAILABLE_CODE_GENERATIVE_MODELS)

    if st.button("Generate Code"):
        generated_code = generate_code(code_idea, selected_model)
        st.code(generated_code, language="python")

    # Automate Build Process
    st.subheader("Automate Build Process")
    if st.button("Automate"):
        agent = AIAgent(selected_agent, "", [])  # Load the agent without skills for now
        summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects, project_name, selected_model)
        st.write("Autonomous Build Summary:")
        st.write(summary)
        st.write("Next Step:")
        st.write(next_step)
    
    # Using the modified and extended class and functions, update the callback for the 'Automate' button in the Streamlit UI:
    if st.button("Automate", args=(hf_token,)):
        agent = AIAgent(selected_agent, "", [])  # Load the agent without skills for now
        summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects, project_name, selected_model, hf_token)
        st.write("Autonomous Build Summary:")
        st.write(summary)
        st.write("Next Step:")
        st.write(next_step)

    # If everything went well, proceed to deploy the Space
    if agent._hf_api and agent.has_valid_hf_token():
        agent.deploy_built_space_to_hf() 
        # Use the hf_token to interact with the Hugging Face API
        api = HfApi(token=hf_token)
        # Function to create a Space on Hugging Face
def create_space(api, name, description, public, files, entrypoint="launch.py"):
    url = f"{hf_hub_url()}spaces/{name}/prepare-repo"
    headers = {"Authorization": f"Bearer {api.access_token}"}