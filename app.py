import streamlit as st
import os
import subprocess
import random
import string
from huggingface_hub import cached_download, hf_hub_url
from transformers import pipeline
import black
import pylint

# Define functions for each feature

# 1. Chat Interface
def chat_interface(input_text):
    """Handles user input in the chat interface.

    Args:
        input_text: User's input text.

    Returns:
        The chatbot's response.
    """
    # Load the appropriate language model from Hugging Face
    model_name = 'google/flan-t5-xl'  # Choose a suitable model
    model_url = hf_hub_url(repo_id=model_name, revision='main', filename='config.json')
    model_path = cached_download(model_url)
    generator = pipeline('text-generation', model=model_path)

    # Generate chatbot response
    response = generator(input_text, max_length=50, num_return_sequences=1, do_sample=True)[0]['generated_text']
    return response

# 2. Terminal
def terminal_interface(command):
    """Executes commands in the terminal.

    Args:
        command: User's command.

    Returns:
        The terminal output.
    """
    # Execute command
    try:
        process = subprocess.run(command.split(), capture_output=True, text=True)
        output = process.stdout
    except Exception as e:
        output = f'Error: {e}'
    return output

# 3. Code Editor
def code_editor_interface(code):
    """Provides code completion, formatting, and linting in the code editor.

    Args:
        code: User's code.

    Returns:
        Formatted and linted code.
    """
    # Format code using black
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
    except black.InvalidInput:
        formatted_code = code  # Keep original code if formatting fails

    # Lint code using pylint
    try:
        pylint_output = pylint.run(formatted_code, output=None)
        lint_results = pylint_output.linter.stats.get('global_note', 0)
        lint_message = f"Pylint score: {lint_results:.2f}"
    except Exception as e:
        lint_message = f"Pylint error: {e}"

    return formatted_code, lint_message

# 4. Workspace
def workspace_interface(project_name):
    """Manages projects, files, and resources in the workspace.

    Args:
        project_name: Name of the new project.

    Returns:
        Project creation status.
    """
    # Create project directory
    try:
        os.makedirs(os.path.join('projects', project_name))
        status = f'Project \"{project_name}\" created successfully.'
    except FileExistsError:
        status = f'Project \"{project_name}\" already exists.'
    return status

# 5. AI-Infused Tools

# Define custom AI-powered tools using Hugging Face models

# Example: Text summarization tool
def summarize_text(text):
    """Summarizes a given text using a Hugging Face model.

    Args:
        text: Text to be summarized.

    Returns:
        Summarized text.
    """
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    summary = summarizer(text, max_length=100, min_length=30)[0]['summary_text']
    return summary

# Streamlit App
st.title("CodeCraft: Your AI-Powered Development Toolkit")

# Chat Interface
st.header("Chat with CodeCraft")
chat_input = st.text_area("Enter your message:")
if st.button("Send"):
    chat_response = chat_interface(chat_input)
    st.write(f"CodeCraft: {chat_response}")

# Terminal Interface
st.header("Terminal")
terminal_input = st.text_input("Enter a command:")
if st.button("Run"):
    terminal_output = terminal_interface(terminal_input)
    st.code(terminal_output, language="bash")

# Code Editor Interface
st.header("Code Editor")
code_editor = st.code_area("Write your code:", language="python")
if st.button("Format & Lint"):
    formatted_code, lint_message = code_editor_interface(code_editor)
    st.code(formatted_code, language="python")
    st.info(lint_message)

# Workspace Interface
st.header("Workspace")
project_name = st.text_input("Enter project name:")
if st.button("Create Project"):
    workspace_status = workspace_interface(project_name)
    st.success(workspace_status)

# AI-Infused Tools
st.header("AI-Powered Tools")
text_to_summarize = st.text_area("Enter text to summarize:")
if st.button("Summarize"):
    summary = summarize_text(text_to_summarize)
    st.write(f"Summary: {summary}")