import os

import streamlit as st
import subprocess
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel, RagRetriever, AutoModelForSeq2SeqLM
import black
from pylint import lint
import sys
import torch
from huggingface_hub import hf_hub_url, cached_download, HfApi
import base64

# Set your Hugging Face API key here
# hf_token = "YOUR_HUGGING_FACE_API_KEY"  # Replace with your actual token
# Get Hugging Face token from secrets.toml - this line should already be in the main code
hf_token = st.secrets["huggingface"]["hf_token"]

HUGGING_FACE_REPO_URL = "https://huggingface.co/spaces/acecalisto3/DevToolKit"
PROJECT_ROOT = "projects"
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