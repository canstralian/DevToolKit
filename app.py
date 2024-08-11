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



# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# User and Project models (as defined earlier)

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

# Authentication routes (as defined earlier)

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
    def __init__(self, plugin_dir):
        self.plugin_dir = plugin_dir
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
# save this as a .py file in your plugin directory
def register_plugin():
    return ExamplePlugin()

class ExamplePlugin:
    name = "example_plugin"

    def run(self, input_data):
        return f"Plugin processed: {input_data}"

plugin_manager = PluginManager('./plugins')
plugin_manager.load_plugins()

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
    # AI assistant code (as before)

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

def deploy_to_hf_spaces(project_name, token):
    hf_api = HfApi()
    repo_url = create_repo(project_name, repo_type="space", space_sdk="streamlit", token=token)
    api.upload_folder(
        folder_path=".",
        repo_id=project_name,
        repo_type="space",
        token=token
    )
    return repo_url

if __name__ == "__main__":
    db.create_all()  # Create the database tables if they don't exist
    main()