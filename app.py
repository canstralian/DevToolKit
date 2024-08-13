import streamlit as st
from flask import Flask, jsonify, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pdb
import subprocess
import docker
from huggingface_hub import HfApi, create_repo
import importlib
import os
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer

   # Load the tokenizer explicitly
   tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py", clean_up_tokenization_spaces=True)

   # Initialize the model
   model = AutoModelForSequenceClassification.from_pretrained("microsoft/CodeGPT-small-py")  # Use a public model

   # Initialize the pipeline
   code_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a strong secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# User and Project models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(100), nullable=False)
    projects = db.relationship('Project', backref='user', lazy=True)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Authentication routes
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists'}), 400
    new_user = User(username=username, password_hash=generate_password_hash(password))
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        login_user(user)
        return jsonify({'message': 'Logged in successfully'}), 200
    return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/create_project', methods=['POST'])
@login_required
def create_project():
    data = request.get_json()
    project_name = data.get('project_name')
    new_project = Project(name=project_name, user_id=current_user.id)
    db.session.add(new_project)
    db.session.commit()
    return jsonify({'message': 'Project created successfully'}), 201

@app.route('/get_projects')
@login_required
def get_projects():
    projects = Project.query.filter_by(user_id=current_user.id).all()
    return jsonify({'projects': [project.name for project in projects]}), 200

# Plugin system
class PluginManager:
    def __init__(self):
        self.plugin_dir = './plugins' 
        self.plugins = {}

    def load_plugins(self):
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('.py'):
                module_name = filename[:-3]
                spec = importlib.util.spec_from_file_location(module_name, os.path.join(self.plugin_dir, filename))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'register_plugin'):
                    plugin = module.register_plugin()
                    self.plugins[plugin.name] = plugin

    def get_plugin(self, name):
        return self.plugins.get(name)

    def list_plugins(self):
        return list(self.plugins.keys())

# Example plugin
def register_plugin():
    return ExamplePlugin()

class ExamplePlugin:
    name = "example_plugin"

    def run(self, input_data):
        return f"Plugin processed: {input_data}"

plugin_manager = PluginManager()
plugin_manager.load_plugins()

# AI Assistant
model = AutoModelForSequenceClassification.from_pretrained("microsoft/CodeGPT-small-py")
codex_pipeline = pipeline("text-generation", model=model)

hf_api = HfApi()

def generate_app(user_idea, project_name):
    # Extract key information from the user idea
    # (You might want to use a more sophisticated NLP pipeline here)
    summary = user_idea  # For now, just use the user's input

    # Create project directory if it doesn't exist
    project_path = create_project(project_name)

    # Generate code using Codex
    prompt = f"""Create a simple Streamlit app for the project named '{project_name}'. The app should display the following summary: '{summary}'."""
    generated_code = codex_pipeline(prompt, max_length=516)[0]['generated_text']

    # Save the generated code to a file in the project directory
    with open(os.path.join(project_path, "app.py"), "w") as f:
        f.write(generated_code)

    # Deploy the app to Hugging Face Spaces
    deploy_app_to_hf_spaces(project_name, generated_code)

    return generated_code, project_path

def deploy_app_to_hf_spaces(project_name, generated_code):
    repo_name = f"hf-{project_name}"
    repo_id = hf_api.changelog.get_repo_id(repo_name)

    if not repo_id:
        create_repo(hf_api, repo_name, "public", "Streamlit App")
        repo_id = hf_api.changelog.get_repo_id(repo_name)

    # Save the generated code to a temporary file
    temp_file = "temp_code.py"
    with open(temp_file, "w") as f:
        f.write(generated_code)

    # Upload the file to Hugging Face Spaces
    hf_api.upload_files(repo_id, [temp_file], hf_api.api_key)

    # Delete the temporary file
    os.remove(temp_file)

    # Print success message
    st.write(f"App deployed successfully to Hugging Face Spaces: https://huggingface.co/spaces/{repo_name}")

def create_project(project_name):
    project_path = os.path.join(os.getcwd(), project_name)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    return project_path

def main():
    st.sidebar.title("AI-Guided Development")
    app_mode = st.sidebar.selectbox("Choose the app mode", 
                                    ["Home", "Login/Register", "File Explorer", "Code Editor", "Terminal", 
                                     "Build & Deploy", "AI Assistant", "Plugins"])

    # AI Guide Toggle
    ai_guide_level = st.sidebar.radio("AI Guide Level", ["Full Assistance", "Partial Assistance", "No Assistance"])

    if app_mode == "Home":
        st.title("Welcome to AI-Guided Development")
        st.write("Select a mode from the sidebar to get started.")

    elif app_mode == "Login/Register":
        login_register_page()

    elif app_mode == "File Explorer":
        file_explorer_page()

    elif app_mode == "Code Editor":
        code_editor_page()

    elif app_mode == "Terminal":
        terminal_page()

    elif app_mode == "Build & Deploy":
        build_and_deploy_page()

    elif app_mode == "AI Assistant":
        ai_assistant_page()

    elif app_mode == "Plugins":
        plugins_page()

@login_required
def file_explorer_page():
    st.header("File Explorer")
    # File explorer code (as before)

@login_required
def code_editor_page():
    st.header("Code Editor")
    # Code editor with Monaco integration
    st.components.v1.html(
        """
        <div id="monaco-editor" style="width:800px;height:600px;border:1px solid grey"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.20.0/min/vs/loader.min.js"></script>
        <script>
            require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.20.0/min/vs' }});
            require(['vs/editor/editor.main'], function() {
                var editor = monaco.editor.create(document.getElementById('monaco-editor'), {
                    value: 'print("Hello, World!")',
                    language: 'python'
                });
            });
        </script>
        """,
        height=650,
    )
    
    if st.button("Run Code"):
        code = st.session_state.get('code', '')  # Get code from Monaco editor
        output = run_code(code)
        st.code(output)
    
    if st.button("Debug Code"):
        code = st.session_state.get('code', '')
        st.write("Debugging mode activated. Check your console for the debugger.")
        debug_code(code)

@login_required
def terminal_page():
    st.header("Terminal")
    # Terminal code (as before)

@login_required
def build_and_deploy_page():
    st.header("Build & Deploy")
    project_name = st.text_input("Enter project name:")
    
    if st.button("Build Docker Image"):
        image, logs = build_docker_image(project_name)
        st.write(f"Docker image built: {image.tags}")
    
    if st.button("Run Docker Container"):
        port = st.number_input("Enter port number:", value=8501)
        container = run_docker_container(project_name, port)
        st.write(f"Docker container running: {container.id}")
    
    if st.button("Deploy to Hugging Face Spaces"):
        token = st.text_input("Enter your Hugging Face token:", type="password")
        if token:
            repo_url = deploy_to_hf_spaces(project_name, token)
            st.write(f"Deployed to Hugging Face Spaces: {repo_url}")

@login_required
def ai_assistant_page():
    st.header("AI Assistant")
    user_idea = st.text_area("Describe your app idea:")
    project_name = st.text_input("Enter project name:")

    if st.button("Generate App"):
        generated_code, project_path = generate_app(user_idea, project_name)
        st.code(generated_code)
        st.write(f"Project directory: {project_path}")

@login_required
def plugins_page():
    st.header("Plugins")
    st.write("Available plugins:")
    for plugin_name in plugin_manager.list_plugins():
        st.write(f"- {plugin_name}")
    
    selected_plugin = st.selectbox("Select a plugin to run:", plugin_manager.list_plugins())
    input_data = st.text_input("Enter input for the plugin:")
    
    if st.button("Run Plugin"):
        plugin = plugin_manager.get_plugin(selected_plugin)
        if plugin:
            result = plugin.run(input_data)
            st.write(f"Plugin output: {result}")

def login_register_page():
    st.header("Login/Register")
    action = st.radio("Choose action:", ["Login", "Register"])
    
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    
    if action == "Login":
        if st.button("Login"):
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
    else:
        if st.button("Register"):
            if User.query.filter_by(username=username).first():
                st.error("Username already exists")
            else:
                new_user = User(username=username, password_hash=generate_password_hash(password))
                db.session.add(new_user)
                db.session.commit()
                st.success("User registered successfully!")

def debug_code(code):
    try:
        pdb.run(code)
    except Exception as e:
        return str(e)

def run_code(code):
    try:
        result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=10)
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Code execution timed out"
    except Exception as e:
        return str(e)

def build_docker_image(project_name):
    client = docker.from_env()
    image, build_logs = client.images.build(path=".", tag=project_name)
    return image, build_logs

def run_docker_container(image_name, port):
    client = docker.from_env()
    container = client.containers.run(image_name, detach=True, ports={f'{port}/tcp': port})
    return container

if __name__ == "__main__":
    db.create_all()  # Create the database tables if they don't exist
    main()