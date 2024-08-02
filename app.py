import subprocess
import os
from io import StringIO
import sys
import black
import streamlit as st
from pylint import lint
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import pipeline as transformers_pipeline
from huggingface_hub import hf_hub_url, cached_download
import json
import time
import shutil
import gradio as gr

# --- Global State ---
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'workspace_projects' not in st.session_state:
    st.session_state.workspace_projects = {}
if 'available_agents' not in st.session_state:
    st.session_state.available_agents = []
if 'available_clusters' not in st.session_state:
    st.session_state.available_clusters = []
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = None
if 'current_cluster' not in st.session_state:
    st.session_state.current_cluster = None
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = None
if 'repo_name' not in st.session_state:
    st.session_state.repo_name = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'selected_code_model' not in st.session_state:
    st.session_state.selected_code_model = None
if 'selected_chat_model' not in st.session_state:
    st.session_state.selected_chat_model = None

# --- Agent Class ---
class AIAgent:
    def __init__(self, name, description, skills, persona_prompt=None):
        self.name = name
        self.description = description
        self.skills = skills
        self.persona_prompt = persona_prompt

    def create_agent_prompt(self):
        skills_str = '\n'.join([f"* {skill}" for skill in self.skills])
        agent_prompt = f"""
I am an AI agent named {self.name}, designed to assist developers with their projects. 
My expertise lies in the following areas:

{skills_str}

{self.persona_prompt if self.persona_prompt else ''}

I am here to help you build, deploy, and improve your applications. 
Feel free to ask me any questions or present me with any challenges you encounter. 
I will do my best to provide helpful and insightful responses.
"""
        return agent_prompt

    def autonomous_build(self, chat_history, workspace_projects):
        """
        Autonomous build logic that continues based on the state of chat history and workspace projects.
        """
        # Example logic: Generate a summary of chat history and workspace state
        summary = "Chat History:\n" + "\n".join([f"User: {u}\nAgent: {a}" for u, a in chat_history])
        summary += "\n\nWorkspace Projects:\n" + "\n".join([f"{p}: {details}" for p, details in workspace_projects.items()])

        # Example: Generate the next logical step in the project
        next_step = "Based on the current state, the next logical step is to implement the main application logic."

        return summary, next_step

# --- Agent Management ---
def save_agent_to_file(agent):
    """Saves the agent's prompt to a file."""
    if not os.path.exists("agents"):
        os.makedirs("agents")
    file_path = os.path.join("agents", f"{agent.name}.txt")
    with open(file_path, "w") as file:
        file.write(agent.create_agent_prompt())
    st.session_state.available_agents.append(agent.name)

def load_agent_prompt(agent_name):
    """Loads an agent prompt from a file."""
    file_path = os.path.join("agents", f"{agent_name}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            agent_prompt = file.read()
        return agent_prompt
    else:
        return None

def create_agent_from_text(name, text, persona_prompt=None):
    skills = text.split('\n')
    agent = AIAgent(name, "AI agent created from text input.", skills, persona_prompt)
    save_agent_to_file(agent)
    return agent.create_agent_prompt()

# --- Cluster Management ---
def create_agent_cluster(cluster_name, agent_names):
    """Creates a cluster of agents."""
    if not os.path.exists("clusters"):
        os.makedirs("clusters")
    cluster_path = os.path.join("clusters", f"{cluster_name}.json")
    with open(cluster_path, "w") as file:
        json.dump({"agents": agent_names}, file)
    st.session_state.available_clusters.append(cluster_name)

def load_agent_cluster(cluster_name):
    """Loads an agent cluster from a file."""
    cluster_path = os.path.join("clusters", f"{cluster_name}.json")
    if os.path.exists(cluster_path):
        with open(cluster_path, "r") as file:
            cluster_data = json.load(file)
        return cluster_data["agents"]
    else:
        return None

# --- Chat Interface ---
def chat_interface_with_agent(input_text, agent_name):
    agent_prompt = load_agent_prompt(agent_name)
    if agent_prompt is None:
        return f"Agent {agent_name} not found."

    # Use a more powerful language model (GPT-3 or similar) for better chat experience
    model_name = st.session_state.selected_chat_model or "text-davinci-003"  # Default to GPT-3 if not selected
    try:
        model = transformers_pipeline("text-generation", model=model_name)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Combine the agent prompt with user input
    combined_input = f"{agent_prompt}\n\nUser: {input_text}\nAgent:"

    # Generate response
    response = model(combined_input, max_length=200, temperature=0.7, top_p=0.95, do_sample=True)[0]['generated_text']
    response = response.split("Agent:")[1].strip()  # Extract the agent's response
    return response

def chat_interface_with_cluster(input_text, cluster_name):
    agent_names = load_agent_cluster(cluster_name)
    if agent_names is None:
        return f"Cluster {cluster_name} not found."

    # Use a more powerful language model (GPT-3 or similar) for better chat experience
    model_name = st.session_state.selected_chat_model or "text-davinci-003"  # Default to GPT-3 if not selected
    try:
        model = transformers_pipeline("text-generation", model=model_name)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Combine the agent prompt with user input
    combined_input = f"User: {input_text}\n"
    for agent_name in agent_names:
        agent_prompt = load_agent_prompt(agent_name)
        combined_input += f"\n{agent_name}:\n{agent_prompt}\n"

    # Generate response
    response = model(combined_input, max_length=200, temperature=0.7, top_p=0.95, do_sample=True)[0]['generated_text']
    response = response.split("User:")[1].strip()  # Extract the agent's response
    return response

# --- Code Editor ---
def code_editor_interface(code):
    """Provides code completion, formatting, and linting in the code editor."""
    # Format code using black
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
    except black.InvalidInput:
        formatted_code = code  # Keep original code if formatting fails

    # Lint code using pylint
    try:
        pylint_output = StringIO()
        sys.stdout = pylint_output
        sys.stderr = pylint_output
        lint.Run(['--from-stdin'], stdin=StringIO(formatted_code))
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        lint_message = pylint_output.getvalue()
    except Exception as e:
        lint_message = f"Pylint error: {e}"

    return formatted_code, lint_message

# --- Workspace Management ---
def workspace_interface(project_name):
    """Manages projects, files, and resources in the workspace."""
    project_path = os.path.join("projects", project_name)
    # Create project directory
    try:
        os.makedirs(project_path)
        requirements_path = os.path.join(project_path, "requirements.txt")
        with open(requirements_path, "w") as req_file:
            req_file.write("")  # Initialize an empty requirements.txt file
        status = f'Project "{project_name}" created successfully.'
        st.session_state.workspace_projects[project_name] = {'files': []}
    except FileExistsError:
        status = f'Project "{project_name}" already exists.'
    return status

def add_code_to_workspace(project_name, code, file_name):
    """Adds selected code files to the workspace."""
    project_path = os.path.join("projects", project_name)
    file_path = os.path.join(project_path, file_name)

    try:
        with open(file_path, "w") as code_file:
            code_file.write(code)
        status = f'File "{file_name}" added to project "{project_name}" successfully.'
        st.session_state.workspace_projects[project_name]['files'].append(file_name)
    except Exception as e:
        status = f"Error: {e}"
    return status

# --- AI Tools ---
def summarize_text(text):
    """Summarizes a given text using a Hugging Face model."""
    model_name = "facebook/bart-large-cnn"
    try:
        summarizer = pipeline("summarization", model=model_name)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Truncate input text to avoid exceeding the model's maximum length
    max_input_length = model.config.max_length
    inputs = text
    if len(text) > max_input_length:
        inputs = text[:max_input_length]

    # Generate summary
    summary = summarizer(inputs, max_length=100, min_length=30, do_sample=False)[0][
        "summary_text"
    ]
    return summary

def sentiment_analysis(text):
    """Performs sentiment analysis on a given text using a Hugging Face model."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    try:
        analyzer = pipeline("sentiment-analysis", model=model_name)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Perform sentiment analysis
    result = analyzer(text)[0]
    return result

def translate_code(code, source_language, target_language):
    """Translates code from one programming language to another using a Hugging Face model."""
    model_name = "Helsinki-NLP/opus-mt-en-fr"  # Replace with your preferred translation model
    try:
        translator = pipeline("translation", model=model_name)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Translate code
    translated_code = translator(code, target_lang=target_language)[0]['translation_text']
    return translated_code

def generate_code(idea):
    """Generates code based on a given idea using a Hugging Face model."""
    model_name = st.session_state.selected_code_model or "bigcode/starcoder"  # Default to Starcoder if not selected
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Generate the code
    input_text = f"""
    # Idea: {idea}
    # Code:
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=model.config.max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7,  # Adjust temperature for creativity
        top_k=50,  # Adjust top_k for diversity
    )
    generated_code = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Remove the prompt and formatting
    parts = generated_code.split("\n# Code:")
    if len(parts) > 1:
        generated_code = parts[1].strip()
    else:
        generated_code = generated_code.strip()

    return generated_code

# --- AI Personas Creator ---
def create_persona_from_text(text):
    """Creates an AI persona from the given text."""
    persona_prompt = f"""
As an elite expert developer with the highest level of proficiency in Streamlit, Gradio, and Hugging Face, I possess a comprehensive understanding of these technologies and their applications in web development and deployment. My expertise encompasses the following areas:

Streamlit:
* In-depth knowledge of Streamlit's architecture, components, and customization options.
* Expertise in creating interactive and user-friendly dashboards and applications.
* Proficiency in integrating Streamlit with various data sources and machine learning models.

Gradio:
* Thorough understanding of Gradio's capabilities for building and deploying machine learning interfaces.
* Expertise in creating custom Gradio components and integrating them with Streamlit applications.
* Proficiency in using Gradio to deploy models from Hugging Face and other frameworks.

Hugging Face:
* Comprehensive knowledge of Hugging Face's model hub and Transformers library.
* Expertise in fine-tuning and deploying Hugging Face models for various NLP and computer vision tasks.
* Proficiency in using Hugging Face's Spaces platform for model deployment and sharing.

Deployment:
* In-depth understanding of best practices for deploying Streamlit and Gradio applications.
* Expertise in deploying models on cloud platforms such as AWS, Azure, and GCP.
* Proficiency in optimizing deployment configurations for performance and scalability.

Additional Skills:
* Strong programming skills in Python and JavaScript.
* Familiarity with Docker and containerization technologies.
* Excellent communication and problem-solving abilities.

I am confident that I can leverage my expertise to assist you in developing and deploying cutting-edge web applications using Streamlit, Gradio, and Hugging Face. Please feel free to ask any questions or present any challenges you may encounter.

Example:

Task:
Develop a Streamlit application that allows users to generate text using a Hugging Face model. The application should include a Gradio component for user input and model prediction.

Solution:

import streamlit as st
import gradio as gr
from transformers import pipeline

# Create a Hugging Face pipeline
huggingface_model = pipeline("text-generation")

# Create a Streamlit app
st.title("Hugging Face Text Generation App")

# Define a Gradio component
demo = gr.Interface(
    fn=huggingface_model,
    inputs=gr.Textbox(lines=2),
    outputs=gr.Textbox(lines=1),
)

# Display the Gradio component in the Streamlit app
st.write(demo)
"""
    return persona_prompt

# --- Terminal Interface ---
def terminal_interface(command, project_name=None):
    """Executes commands in the terminal."""
    # Execute command
    try:
        process = subprocess.run(command.split(), capture_output=True, text=True)
        output = process.stdout

        # If the command is to install a package, update the workspace
        if "install" in command and project_name:
            requirements_path = os.path.join("projects", project_name, "requirements.txt")
            with open(requirements_path, "a") as req_file:
                package_name = command.split()[-1]
                req_file.write(f"{package_name}\n")
    except Exception as e:
        output = f"Error: {e}"
    return output

# --- Build and Deploy ---
def build_project(project_name):
    """Builds a project based on the workspace files."""
    project_path = os.path.join("projects", project_name)
    requirements_path = os.path.join(project_path, "requirements.txt")
    
    # Install dependencies
    os.chdir(project_path)
    terminal_interface(f"pip install -r {requirements_path}")
    os.chdir("..")

    # Create a temporary directory for the built project
    build_dir = os.path.join("build", project_name)
    os.makedirs(build_dir, exist_ok=True)

    # Copy project files to the build directory
    for filename in os.listdir(project_path):
        if filename == "requirements.txt":
            continue
        shutil.copy(os.path.join(project_path, filename), build_dir)

    # Create a `main.py` file if it doesn't exist
    main_file = os.path.join(build_dir, "main.py")
    if not os.path.exists(main_file):
        with open(main_file, "w") as f:
            f.write("# Your Streamlit app code goes here\n")

    # Return the path to the built project
    return build_dir

def deploy_to_huggingface(build_dir, hf_token, repo_name):
    """Deploys the built project to Hugging Face Spaces."""
    # Authenticate with Hugging Face
    os.environ["HF_TOKEN"] = hf_token

    # Create a new Hugging Face Space repository
    try:
        subprocess.run(f"huggingface-cli repo create {repo_name}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error creating Hugging Face Space repository: {e}")
        return

    # Upload the built project to the repository
    try:
        subprocess.run(f"huggingface-cli upload {repo_name} {build_dir}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error uploading project to Hugging Face Space repository: {e}")
        return

    # Deploy the project to Hugging Face Spaces
    try:
        subprocess.run(f"huggingface-cli space deploy {repo_name}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error deploying project to Hugging Face Spaces: {e}")
        return

    # Display the deployment URL
    st.success(f"Project deployed successfully to Hugging Face Spaces: https://huggingface.co/spaces/{repo_name}")

def deploy_locally(build_dir):
    """Deploys the built project locally."""
    # Run the project locally
    os.chdir(build_dir)
    subprocess.run("streamlit run main.py", shell=True, check=True)
    os.chdir("..")

    # Display a success message
    st.success(f"Project deployed locally!")

# --- Streamlit App ---
st.set_page_config(page_title="AI Agent Creator", page_icon="🤖")

# --- Tabs for Navigation ---
tabs = st.tabs(["AI Agent Creator", "Tool Box", "Workspace Chat App"])

# --- AI Agent Creator ---
with tabs[0]:
    st.header("Create an AI Agent from Text")

    st.subheader("From Text")
    agent_name = st.text_input("Enter agent name:")
    text_input = st.text_area("Enter skills (one per line):")
    persona_prompt_option = st.selectbox("Choose a persona prompt", ["None", "Expert Developer"])
    persona_prompt = None
    if persona_prompt_option == "Expert Developer":
        persona_prompt = create_persona_from_text("Expert Developer")
    if st.button("Create Agent"):
        agent_prompt = create_agent_from_text(agent_name, text_input, persona_prompt)
        st.success(f"Agent '{agent_name}' created and saved successfully.")
        st.session_state.available_agents.append(agent_name)

    st.subheader("Create an Agent Cluster")
    cluster_name = st.text_input("Enter cluster name:")
    agent_names = st.multiselect("Select agents for the cluster", st.session_state.available_agents)
    if st.button("Create Cluster"):
        create_agent_cluster(cluster_name, agent_names)
        st.success(f"Cluster '{cluster_name}' created successfully.")
        st.session_state.available_clusters.append(cluster_name)

# --- Tool Box ---
with tabs[1]:
    st.header("Tool Box")

    # --- Workspace ---
    st.subheader("Workspace")
    project_name = st.selectbox("Select a project", list(st.session_state.workspace_projects.keys()), key="project_select")
    if project_name:
        st.session_state.current_project = project_name
        for file in st.session_state.workspace_projects[project_name]['files']:
            st.write(f"  - {file}")

    # --- Chat with AI Agents ---
    st.subheader("Chat with AI Agents")
    selected_agent_or_cluster = st.selectbox("Select an AI agent or cluster", st.session_state.available_agents + st.session_state.available_clusters)
    agent_chat_input = st.text_area("Enter your message:")
    chat_model_options = ["text-davinci-003", "gpt-3.5-turbo"]  # Add more chat models as needed
    selected_chat_model = st.selectbox("Select a chat model", chat_model_options)
    if st.button("Send"):
        st.session_state.selected_chat_model = selected_chat_model
        if selected_agent_or_cluster in st.session_state.available_agents:
            st.session_state.current_agent = selected_agent_or_cluster
            st.session_state.current_cluster = None
            agent_chat_response = chat_interface_with_agent(agent_chat_input, selected_agent_or_cluster)
        elif selected_agent_or_cluster in st.session_state.available_clusters:
            st.session_state.current_agent = None
            st.session_state.current_cluster = selected_agent_or_cluster
            agent_chat_response = chat_interface_with_cluster(agent_chat_input, selected_agent_or_cluster)
        else:
            agent_chat_response = "Invalid selection."
        st.session_state.chat_history.append((agent_chat_input, agent_chat_response))
        st.write(f"{selected_agent_or_cluster}: {agent_chat_response}")

    # --- Automate Build Process ---
    st.subheader("Automate Build Process")
    if st.button("Automate"):
        if st.session_state.current_agent:
            agent = AIAgent(st.session_state.current_agent, "", [])  # Load the agent without skills for now
            summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects)
            st.write("Autonomous Build Summary:")
            st.write(summary)
            st.write("Next Step:")
            st.write(next_step)
        elif st.session_state.current_cluster:
            # Implement cluster-based automation logic here
            # ...
            st.warning("Cluster-based automation is not yet implemented.")
        else:
            st.warning("Please select an agent or cluster first.")

# --- Workspace Chat App ---
with tabs[2]:
    st.header("Workspace Chat App")

    # --- Project Selection ---
    project_name = st.selectbox("Select a project", list(st.session_state.workspace_projects.keys()), key="project_select")
    if project_name:
        st.session_state.current_project = project_name

    # --- Chat with AI Agents ---
    st.subheader("Chat with AI Agents")
    selected_agent_or_cluster = st.selectbox("Select an AI agent or cluster", st.session_state.available_agents + st.session_state.available_clusters)
    agent_chat_input = st.text_area("Enter your message:")
    chat_model_options = ["text-davinci-003", "gpt-3.5-turbo"]  # Add more chat models as needed
    selected_chat_model = st.selectbox("Select a chat model", chat_model_options)
    if st.button("Send"):
        st.session_state.selected_chat_model = selected_chat_model
        if selected_agent_or_cluster in st.session_state.available_agents:
            st.session_state.current_agent = selected_agent_or_cluster
            st.session_state.current_cluster = None
            agent_chat_response = chat_interface_with_agent(agent_chat_input, selected_agent_or_cluster)
        elif selected_agent_or_cluster in st.session_state.available_clusters:
            st.session_state.current_agent = None
            st.session_state.current_cluster = selected_agent_or_cluster
            agent_chat_response = chat_interface_with_cluster(agent_chat_input, selected_agent_or_cluster)
        else:
            agent_chat_response = "Invalid selection."
        st.session_state.chat_history.append((agent_chat_input, agent_chat_response))
        st.write(f"{selected_agent_or_cluster}: {agent_chat_response}")

    # --- Code Editor ---
    st.subheader("Code Editor")
    code = st.text_area("Enter your code:")
    if st.button("Format & Lint"):
        formatted_code, lint_message = code_editor_interface(code)
        st.code(formatted_code, language="python")
        st.write("Linting Report:")
        st.write(lint_message)

    # --- Add Code to Workspace ---
    st.subheader("Add Code to Workspace")
    file_name = st.text_input("Enter file name:")
    if st.button("Add Code"):
        if st.session_state.current_project:
            status = add_code_to_workspace(st.session_state.current_project, code, file_name)
            st.write(status)
        else:
            st.warning("Please select a project first.")

    # --- Terminal ---
    st.subheader("Terminal")
    command = st.text_input("Enter a command:")
    if st.button("Execute"):
        if st.session_state.current_project:
            output = terminal_interface(command, st.session_state.current_project)
            st.write(output)
        else:
            st.warning("Please select a project first.")

    # --- AI Tools ---
    st.subheader("AI Tools")
    st.write("Summarize Text:")
    text_to_summarize = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        summary = summarize_text(text_to_summarize)
        st.write(summary)

    st.write("Sentiment Analysis:")
    text_to_analyze = st.text_area("Enter text to analyze:")
    if st.button("Analyze"):
        result = sentiment_analysis(text_to_analyze)
        st.write(result)

    st.write("Code Translation:")
    code_to_translate = st.text_area("Enter code to translate:")
    source_language = st.selectbox("Source Language", ["Python", "JavaScript", "C++"])
    target_language = st.selectbox("Target Language", ["Python", "JavaScript", "C++"])
    if st.button("Translate"):
        translated_code = translate_code(code_to_translate, source_language, target_language)
        st.write(translated_code)

    st.write("Code Generation:")
    code_idea = st.text_input("Enter your code idea:")
    code_model_options = ["bigcode/starcoder", "google/flan-t5-xl"]  # Add more code models as needed
    selected_code_model = st.selectbox("Select a code generation model", code_model_options)
    if st.button("Generate"):
        st.session_state.selected_code_model = selected_code_model
        generated_code = generate_code(code_idea)
        st.code(generated_code, language="python")

    # --- Build and Deploy ---
    st.subheader("Build and Deploy")
    if st.session_state.current_project:
        st.write(f"Current project: {st.session_state.current_project}")
        if st.button("Build"):
            build_dir = build_project(st.session_state.current_project)
            st.write(f"Project built successfully! Build directory: {build_dir}")

        st.write("Select a deployment target:")
        deployment_target = st.selectbox("Deployment Target", ["Local", "Hugging Face Spaces"])
        if deployment_target == "Hugging Face Spaces":
            hf_token = st.text_input("Enter your Hugging Face token:")
            repo_name = st.text_input("Enter your Hugging Face Space repository name:")
            if st.button("Deploy to Hugging Face Spaces"):
                st.session_state.hf_token = hf_token
                st.session_state.repo_name = repo_name
                deploy_to_huggingface(build_dir, hf_token, repo_name)
        elif deployment_target == "Local":
            if st.button("Deploy Locally"):
                deploy_locally(build_dir)
    else:
        st.warning("Please select a project first.")

# --- Hugging Face Space Deployment (After Building) ---
if st.session_state.hf_token and st.session_state.repo_name:
    st.write("Deploying to Hugging Face Spaces...")
    deploy_to_huggingface(build_dir, st.session_state.hf_token, st.session_state.repo_name)