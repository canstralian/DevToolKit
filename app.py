import streamlit as st
import os
import subprocess
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, HfApi

# Set your Hugging Face API key here
hf_token = "YOUR_HUGGING_FACE_API_KEY"  # Replace with your actual token

PROJECT_ROOT = "projects"
AGENT_DIRECTORY = "agents"
AVAILABLE_CODE_GENERATIVE_MODELS = ["bigcode/starcoder", "Salesforce/codegen-350M-mono", "microsoft/CodeGPT-small"]

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

    def autonomous_build(self, chat_history, workspace_projects, project_name, selected_model, hf_token):
        """
        Autonomous build logic that continues based on the state of chat history and workspace projects.
        """
        # Example logic: Generate a summary of chat history and workspace state
        summary = "Chat History:\n" + "\n".join([f"User: {u}\nAgent: {a}" for u, a in chat_history])
        summary += "\n\nWorkspace Projects:\n" + "\n".join([f"{p}: {details}" for p, details in workspace_projects.items()])

        # Example: Generate the next logical step in the project
        next_step = "Based on the current state, the next logical step is to implement the main application logic."

        return summary, next_step

    def deploy_built_space_to_hf(self):
        # Implement deployment logic here
        pass

def process_input(input_text):
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium", tokenizer="microsoft/DialoGPT-medium")
    response = chatbot(input_text, max_length=50, num_return_sequences=1)[0]['generated_text']
    return response

def run_code(code):
    try:
        result = subprocess.run(code, shell=True, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

def workspace_interface(project_name):
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
        st.session_state.workspace_projects[project_name] = {'files': []}
        return f"Project '{project_name}' created successfully."
    else:
        return f"Project '{project_name}' already exists."

def add_code_to_workspace(project_name, code, file_name):
    project_path = os.path.join(PROJECT_ROOT, project_name)
    if not os.path.exists(project_path):
        return f"Project '{project_name}' does not exist."
    
    file_path = os.path.join(project_path, file_name)
    with open(file_path, "w") as file:
        file.write(code)
    st.session_state.workspace_projects[project_name]['files'].append(file_name)
    return f"Code added to '{file_name}' in project '{project_name}'."

def display_chat_history(chat_history):
    return "\n".join([f"User: {u}\nAgent: {a}" for u, a in chat_history])

def display_workspace_projects(workspace_projects):
    return "\n".join([f"{p}: {details}" for p, details in workspace_projects.items()])

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["AI Agent Creator", "Tool Box", "Workspace Chat App"])

    if app_mode == "AI Agent Creator":
        # AI Agent Creator
        st.header("Create an AI Agent from Text")

        st.subheader("From Text")
        agent_name = st.text_input("Enter agent name:")
        text_input = st.text_area("Enter skills (one per line):")
        if st.button("Create Agent"):
            skills = text_input.split('\n')
            agent = AIAgent(agent_name, "AI agent created from text input", skills)
            st.success(f"Agent '{agent_name}' created and saved successfully.")
            st.session_state.available_agents.append(agent_name)

    elif app_mode == "Tool Box":
        # Tool Box
        st.header("AI-Powered Tools")

        # Chat Interface
        st.subheader("Chat with CodeCraft")
        chat_input = st.text_area("Enter your message:")
        if st.button("Send"):
            response = process_input(chat_input)
            st.session_state.chat_history.append((chat_input, response))
            st.write(f"CodeCraft: {response}")

        # Terminal Interface
        st.subheader("Terminal")
        terminal_input = st.text_input("Enter a command:")
        if st.button("Run"):
            output = run_code(terminal_input)
            st.session_state.terminal_history.append((terminal_input, output))
            st.code(output, language="bash")

        # Project Management
        st.subheader("Project Management")
        project_name_input = st.text_input("Enter Project Name:")
        if st.button("Create Project"):
            status = workspace_interface(project_name_input)
            st.write(status)

        code_to_add = st.text_area("Enter Code to Add to Workspace:", height=150)
        file_name_input = st.text_input("Enter File Name (e.g., 'app.py'):")
        if st.button("Add Code"):
            status = add_code_to_workspace(project_name_input, code_to_add, file_name_input)
            st.write(status)

        # Display Chat History
        st.subheader("Chat History")
        chat_history = display_chat_history(st.session_state.chat_history)
        st.text_area("Chat History", value=chat_history, height=200)

        # Display Workspace Projects
        st.subheader("Workspace Projects")
        workspace_projects = display_workspace_projects(st.session_state.workspace_projects)
        st.text_area("Workspace Projects", value=workspace_projects, height=200)

    elif app_mode == "Workspace Chat App":
        # Workspace Chat App
        st.header("Workspace Chat App")

        # Chat Interface with AI Agents
        st.subheader("Chat with AI Agents")
        selected_agent = st.selectbox("Select an AI agent", st.session_state.available_agents)
        agent_chat_input = st.text_area("Enter your message for the agent:")
        if st.button("Send to Agent"):
            response = process_input(agent_chat_input)
            st.session_state.chat_history.append((agent_chat_input, response))
            st.write(f"{selected_agent}: {response}")

        # Code Generation
        st.subheader("Code Generation")
        code_idea = st.text_input("Enter your code idea:")
        selected_model = st.selectbox("Select a code-generative model", AVAILABLE_CODE_GENERATIVE_MODELS)
        if st.button("Generate Code"):
            generator = pipeline("text-generation", model=selected_model, tokenizer=selected_model)
            generated_code = generator(code_idea, max_length=150, num_return_sequences=1)[0]['generated_text']
            st.code(generated_code, language="python")

        # Automate Build Process
        st.subheader("Automate Build Process")
        if st.button("Automate"):
            agent = AIAgent(selected_agent, "", [])  # Load the agent without skills for now
            summary, next_step = agent.autonomous_build(st.session_state.chat_history, st.session_state.workspace_projects, project_name_input, selected_model, hf_token)
            st.write("Autonomous Build Summary:")
            st.write(summary)
            st.write("Next Step:")
            st.write(next_step)

            if agent._hf_api and agent.has_valid_hf_token():
                repository = agent.deploy_built_space_to_hf()
                st.markdown("## Congratulations! Successfully deployed Space 🚀 ##")
                st.markdown("[Check out your new Space here](hf.co/" + repository.name + ")")

# Launch the Streamlit app
st.markdown("""
<style>
/* Advanced and Accommodating CSS */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f4f4f9;
    color: #333;
    margin: 0;
    padding: 0;
}

h1, h2, h3, h4, h5, h6 {
    color: #333;
}

.container {
    width: 90%;
    margin: 0 auto;
    padding: 20px;
}

/* Navigation Sidebar */
.sidebar {
    background-color: #2c3e50;
    color: #ecf0f1;
    padding: 20px;
    height: 100vh;
    position: fixed;
    top: 0;
    left: 0;
    width: 250px;
    overflow-y: auto;
}

.sidebar a {
    color: #ecf0f1;
    text-decoration: none;
    display: block;
    padding: 10px 0;
}

.sidebar a:hover {
    background-color: #34495e;
    border-radius: 5px;
}

/* Main Content */
.main-content {
    margin-left: 270px;
    padding: 20px;
}

/* Buttons */
button {
    background-color: #3498db;
    color: #fff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    background-color: #2980b9;
}

/* Text Areas and Inputs */
textarea, input[type="text"] {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border: 1px solid #ddd;
    border-radius: 5px;
    box-sizing: border-box;
}

textarea:focus, input[type="text"]:focus {
    border-color: #3498db;
    outline: none;
}

/* Terminal Output */
.code-output {
    background-color: #1e1e1e;
    color: #dcdcdc;
    padding: 20px;
    border-radius: 5px;
    font-family: 'Courier New', Courier, monospace;
}

/* Chat History */
.chat-history {
    background-color: #ecf0f1;
    padding: 20px;
    border-radius: 5px;
    max-height: 300px;
    overflow-y: auto;
}

.chat-message {
    margin-bottom: 10px;
}

.chat-message.user {
    text-align: right;
    color: #3498db;
}

.chat-message.agent {
    text-align: left;
    color: #e74c3c;
}

/* Project Management */
.project-list {
    background-color: #ecf0f1;
    padding: 20px;
    border-radius: 5px;
    max-height: 300px;
    overflow-y: auto;
}

.project-item {
    margin-bottom: 10px;
}

.project-item a {
    color: #3498db;
    text-decoration: none;
}

.project-item a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)