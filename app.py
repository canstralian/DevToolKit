from huggingface_hub import InferenceClient, hf_hub_url
import gradio as gr
import random
import os
import subprocess
import threading
import time
import shutil
from typing import Dict, Tuple, List
import json
from rich import print as rprint
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.traceback import install
install()  # Enable rich tracebacks for easier debugging

# --- Constants ---

API_URL = "https://api-inference.huggingface.co/models/"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Replace with your desired model

# Chat Interface Parameters
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.2

# Local Server
LOCAL_HOST_PORT = 7860

# --- Agent Roles ---

agent_roles: Dict[str, Dict[str, bool]] = {
    "Web Developer": {"description": "A master of front-end and back-end web development.", "active": False},
    "Prompt Engineer": {"description": "An expert in crafting effective prompts for AI models.", "active": False},
    "Python Code Developer": {"description": "A skilled Python programmer who can write clean and efficient code.", "active": False},
    "Hugging Face Hub Expert": {"description": "A specialist in navigating and utilizing the Hugging Face Hub.", "active": False},
    "AI-Powered Code Assistant": {"description": "An AI assistant that can help with coding tasks and provide code snippets.", "active": False},
}

# --- Initial Prompt ---

selected_agent = list(agent_roles.keys())[0]
initial_prompt = f"""
You are an expert {selected_agent} who responds with complete program coding to client requests. 
Using available tools, please explain the researched information.
Please don't answer based solely on what you already know. Always perform a search before providing a response.
In special cases, such as when the user specifies a page to read, there's no need to search.
Please read the provided page and answer the user's question accordingly.
If you find that there's not much information just by looking at the search results page, consider these two options and try them out:
- Try clicking on the links of the search results to access and read the content of each page.
- Change your search query and perform a new search.
Users are extremely busy and not as free as you are.
Therefore, to save the user's effort, please provide direct answers.
BAD ANSWER EXAMPLE
- Please refer to these pages.
- You can write code referring these pages.
- Following page will be helpful.
GOOD ANSWER EXAMPLE
- This is the complete code:  -- complete code here --
- The answer of you question is -- answer here --
Please make sure to list the URLs of the pages you referenced at the end of your answer. (This will allow users to verify your response.)
Please make sure to answer in the language used by the user. If the user asks in Japanese, please answer in Japanese. If the user asks in Spanish, please answer in Spanish.
But, you can go ahead and search in English, especially for programming-related questions. PLEASE MAKE SURE TO ALWAYS SEARCH IN ENGLISH FOR THOSE.
"""

# --- Custom CSS ---

customCSS = """
#component-7 { 
  height: 1600px; 
  flex-grow: 4;
}
"""

# --- Functions ---

# Function to toggle the active state of an agent
def toggle_agent(agent_name: str) -> str:
    """Toggles the active state of an agent."""
    global agent_roles
    agent_roles[agent_name]["active"] = not agent_roles[agent_name]["active"]
    return f"{agent_name} is now {'active' if agent_roles[agent_name]['active'] else 'inactive'}"

# Function to get the active agent cluster
def get_agent_cluster() -> Dict[str, bool]:
    """Returns a dictionary of active agents."""
    return {agent: agent_roles[agent]["active"] for agent in agent_roles}

# Function to execute code
def run_code(code: str) -> str:
    """Executes the provided code and returns the output."""
    try:
        output = subprocess.check_output(
            ['python', '-c', code],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        return output
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output}"

# Function to format the prompt
def format_prompt(message: str, history: list[Tuple[str, str]], agent_roles: list[str]) -> str:
    """Formats the prompt with the selected agent roles and conversation history."""
    prompt = f"""
You are an expert agent cluster, consisting of {', '.join(agent_roles)}. 
Respond with complete program coding to client requests. 
Using available tools, please explain the researched information.
Please don't answer based solely on what you already know. Always perform a search before providing a response.
In special cases, such as when the user specifies a page to read, there's no need to search.
Please read the provided page and answer the user's question accordingly.
If you find that there's not much information just by looking at the search results page, consider these two options and try them out:
- Try clicking on the links of the search results to access and read the content of each page.
- Change your search query and perform a new search.
Users are extremely busy and not as free as you are.
Therefore, to save the user's effort, please provide direct answers.
BAD ANSWER EXAMPLE
- Please refer to these pages.
- You can write code referring these pages.
- Following page will be helpful.
GOOD ANSWER EXAMPLE
- This is the complete code:  -- complete code here --
- The answer of you question is -- answer here --
Please make sure to list the URLs of the pages you referenced at the end of your answer. (This will allow users to verify your response.)
Please make sure to answer in the language used by the user. If the user asks in Japanese, please answer in Japanese. If the user asks in Spanish, please answer in Spanish.
But, you can go ahead and search in English, especially for programming-related questions. PLEASE MAKE SURE TO ALWAYS SEARCH IN ENGLISH FOR THOSE.
"""

    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    
    prompt += f"[INST] {message} [/INST]"
    return prompt

# Function to generate a response
def generate(prompt: str, history: list[Tuple[str, str]], agent_roles: list[str], temperature: float = DEFAULT_TEMPERATURE, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, top_p: float = DEFAULT_TOP_P, repetition_penalty: float = DEFAULT_REPETITION_PENALTY) -> str:
    """Generates a response using the selected agent roles and parameters."""
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=random.randint(0, 10**7),
    )

    formatted_prompt = format_prompt(prompt, history, agent_roles)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output

# Function to handle user input and generate responses
def chat_interface(message: str, history: list[Tuple[str, str]], agent_cluster: Dict[str, bool], temperature: float, max_new_tokens: int, top_p: float, repetition_penalty: float) -> Tuple[str, str]:
    """Handles user input and generates responses."""
    rprint(f"[bold blue]User:[/bold blue] {message}")  # Log user message
    if message.startswith("python"): 
        # User entered code, execute it 
        code = message[9:-3] 
        output = run_code(code) 
        rprint(f"[bold green]Code Output:[/bold green] {output}")  # Log code output
        return (message, output) 
    else:
        # User entered a normal message, generate a response
        active_agents = [agent for agent, is_active in agent_cluster.items() if is_active]
        response = generate(message, history, active_agents, temperature, max_new_tokens, top_p, repetition_penalty)
        rprint(f"[bold purple]Agent Response:[/bold purple] {response}")  # Log agent response
        return (message, response)

# Function to create a new web app instance
def create_web_app(app_name: str, code: str) -> None:
    """Creates a new web app instance with the given name and code."""
    # Create a new directory for the app
    os.makedirs(app_name, exist_ok=True)

    # Create the app.py file
    with open(os.path.join(app_name, 'app.py'), 'w') as f:
        f.write(code)

    # Create the requirements.txt file
    with open(os.path.join(app_name, 'requirements.txt'), 'w') as f:
        f.write("gradio\nhuggingface_hub")

    # Print a success message
    print(f"Web app '{app_name}' created successfully!")

# Function to handle the "Create Web App" button click
def create_web_app_button_click(code: str) -> str:
    """Handles the "Create Web App" button click."""
    # Get the app name from the user
    app_name = gr.Textbox.get().strip()

    # Validate the app name
    if not app_name:
        return "Please enter a valid app name."

    # Create the web app instance
    create_web_app(app_name, code)

    # Return a success message
    return f"Web app '{app_name}' created successfully!"

# Function to handle the "Deploy" button click
def deploy_button_click(app_name: str, code: str) -> str:
    """Handles the "Deploy" button click."""
    # Get the app name from the user
    app_name = gr.Textbox.get().strip()

    # Validate the app name
    if not app_name:
        return "Please enter a valid app name."

    # Get Hugging Face token
    hf_token = gr.Textbox.get("hf_token").strip()

    # Validate Hugging Face token
    if not hf_token:
        return "Please enter a valid Hugging Face token."

    # Create a new directory for the app
    os.makedirs(app_name, exist_ok=True)

    # Copy the code to the app directory
    with open(os.path.join(app_name, 'app.py'), 'w') as f:
        f.write(code)

    # Create the requirements.txt file
    with open(os.path.join(app_name, 'requirements.txt'), 'w') as f:
        f.write("gradio\nhuggingface_hub")

    # Deploy the app to Hugging Face Spaces
    try:
        subprocess.run(
            ['huggingface-cli', 'login', '--token', hf_token],
            check=True,
        )
        subprocess.run(
            ['huggingface-cli', 'space', 'create', app_name, '--repo_type', 'spaces', '--private', '--branch', 'main'],
            check=True,
        )
        subprocess.run(
            ['git', 'init'],
            cwd=app_name,
            check=True,
        )
        subprocess.run(
            ['git', 'add', '.'],
            cwd=app_name,
            check=True,
        )
        subprocess.run(
            ['git', 'commit', '-m', 'Initial commit'],
            cwd=app_name,
            check=True,
        )
        subprocess.run(
            ['git', 'remote', 'add', 'origin', hf_hub_url(username='your_username', repo_id=app_name)],
            cwd=app_name,
            check=True,
        )
        subprocess.run(
            ['git', 'push', '-u', 'origin', 'main'],
            cwd=app_name,
            check=True,
        )
        return f"Web app '{app_name}' deployed successfully to Hugging Face Spaces!"
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"

# Function to handle the "Local Host" button click
def local_host_button_click(app_name: str, code: str) -> str:
    """Handles the "Local Host" button click."""
    # Get the app name from the user
    app_name = gr.Textbox.get().strip()

    # Validate the app name
    if not app_name:
        return "Please enter a valid app name."

    # Create a new directory for the app
    os.makedirs(app_name, exist_ok=True)

    # Copy the code to the app directory
    with open(os.path.join(app_name, 'app.py'), 'w') as f:
        f.write(code)

    # Create the requirements.txt file
    with open(os.path.join(app_name, 'requirements.txt'), 'w') as f:
        f.write("gradio\nhuggingface_hub")

    # Start the local server
    os.chdir(app_name)
    subprocess.Popen(['gradio', 'run', 'app.py', '--share', '--server_port', str(LOCAL_HOST_PORT)])

    # Return a success message
    return f"Web app '{app_name}' running locally on port {LOCAL_HOST_PORT}!"

# Function to handle the "Ship" button click
def ship_button_click(app_name: str, code: str) -> str:
    """Handles the "Ship" button click."""
    # Get the app name from the user
    app_name = gr.Textbox.get().strip()

    # Validate the app name
    if not app_name:
        return "Please enter a valid app name."

    # Ship the web app instance
    # ... (Implement shipping logic here)

    # Return a success message
    return f"Web app '{app_name}' shipped successfully!"

# --- Gradio Interface ---

with gr.Blocks(theme='ParityError/Interstellar') as demo:
    # --- Agent Selection ---
    with gr.Row():
        for agent_name, agent_data in agent_roles.items():
            button = gr.Button(agent_name, variant="secondary")
            textbox = gr.Textbox(agent_data["description"], interactive=False)
            button.click(toggle_agent, inputs=[button], outputs=[textbox])

    # --- Chat Interface ---
    with gr.Row():
        chatbot = gr.Chatbot()
        chat_interface_input = gr.Textbox(label="Enter your message", placeholder="Ask me anything!")
        chat_interface_output = gr.Textbox(label="Response", interactive=False)

        # Parameters
        temperature_slider = gr.Slider(
            label="Temperature",
            value=DEFAULT_TEMPERATURE,
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            interactive=True,
            info="Higher values generate more diverse outputs",
        )
        max_new_tokens_slider = gr.Slider(
            label="Maximum New Tokens",
            value=DEFAULT_MAX_NEW_TOKENS,
            minimum=64,
            maximum=4096,
            step=64,
            interactive=True,
            info="The maximum number of new tokens",
        )
        top_p_slider = gr.Slider(
            label="Top-p (Nucleus Sampling)",
            value=DEFAULT_TOP_P,
            minimum=0.0,
            maximum=1,
            step=0.05,
            interactive=True,
            info="Higher values sample more low-probability tokens",
        )
        repetition_penalty_slider = gr.Slider(
            label="Repetition Penalty",
            value=DEFAULT_REPETITION_PENALTY,
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            interactive=True,
            info="Penalize repeated tokens",
        )

        # Submit Button
        submit_button = gr.Button("Submit")

        # Chat Interface Logic
        submit_button.click(
            chat_interface,
            inputs=[
                chat_interface_input,
                chatbot,
                get_agent_cluster,
                temperature_slider,
                max_new_tokens_slider,
                top_p_slider,
                repetition_penalty_slider,
            ],
            outputs=[
                chatbot,
                chat_interface_output,
            ],
        )

    # --- Web App Creation ---
    with gr.Row():
        app_name_input = gr.Textbox(label="App Name", placeholder="Enter your app name")
        code_output = gr.Textbox(label="Code", interactive=False)
        create_web_app_button = gr.Button("Create Web App")
        deploy_button = gr.Button("Deploy")
        local_host_button = gr.Button("Local Host")
        ship_button = gr.Button("Ship")
        hf_token_input = gr.Textbox(label="Hugging Face Token", placeholder="Enter your Hugging Face token")

        # Web App Creation Logic
        create_web_app_button.click(
            create_web_app_button_click,
            inputs=[code_output],
            outputs=[gr.Textbox(label="Status", interactive=False)],
        )

        # Deploy the web app
        deploy_button.click(
            deploy_button_click,
            inputs=[app_name_input, code_output, hf_token_input],
            outputs=[gr.Textbox(label="Status", interactive=False)],
        )

        # Local host the web app
        local_host_button.click(
            local_host_button_click,
            inputs=[app_name_input, code_output],
            outputs=[gr.Textbox(label="Status", interactive=False)],
        )

        # Ship the web app
        ship_button.click(
            ship_button_click,
            inputs=[app_name_input, code_output],
            outputs=[gr.Textbox(label="Status", interactive=False)],
        )

    # --- Connect Chat Output to Code Output ---
    chat_interface_output.change(
        lambda x: x,
        inputs=[chat_interface_output],
        outputs=[code_output],
    )

    # --- Initialize Hugging Face Client ---
    client = InferenceClient(repo_id=MODEL_NAME, token=os.environ.get("HF_TOKEN"))

    # --- Launch Gradio ---
    demo.queue().launch(debug=True)