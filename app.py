import os
import subprocess
import streamlit as st
from transformers.pipelines import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, RagRetriever, AutoModelForSeq2SeqLM
import black
from pylint import lint
from io import StringIO
import sys
import torch
from huggingface_hub import hf_hub_url, cached_download, HfApi
from datetime import datetime
import requests
import random
from huggingface_hub.hf_api import Repository  # Assuming this is how you import the Repository class

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

# ... (Rest of your existing functions: save_agent_to_file, load_agent_prompt, 
# create_agent_from_text, chat_interface_with_agent, terminal_interface, 
# code_editor_interface, summarize_text, sentiment_analysis, translate_code, 
# generate_code, chat_interface, workspace_interface, add_code_to_workspace) 

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

# ... (Rest of your Streamlit app logic, including the 'Automate' button callback)

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

# ... (Rest of your Streamlit app logic for other app modes)

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