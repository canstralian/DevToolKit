import streamlit as st
import os
import subprocess
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
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
    # Load the GPT-2 model which is compatible with AutoModelForCausalLM
    model_name = 'gpt2'
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    except EnvironmentError as e:
        return f'Error loading model: {e}'

    # Truncate input text to avoid exceeding the model's maximum length
    max_input_length = 900
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    if input_ids.shape[1] > max_input_length:
        input_ids = input_ids[:, :max_input_length]

    # Generate chatbot response
    outputs = model.generate(input_ids, max_new_tokens=50, num_return_sequences=1, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

# 6. Code Generation
def generate_code(idea):
    """Generates code based on a given idea using the EleutherAI/gpt-neo-2.7B model.

    Args:
        idea: The idea for the code to be generated.

    Returns:
        The generated code as a string.
    """

    # Load the code generation model
    model_name = 'EleutherAI/gpt-neo-2.7B'
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except EnvironmentError as e:
        return f'Error loading model: {e}'

    # Generate the code
    input_text = f"""
    # Idea: {idea}
    # Code:
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=1024,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7,  # Adjust temperature for creativity
        top_k=50,  # Adjust top_k for diversity
    )
    generated_code = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Remove the prompt and formatting
    generated_code = generated_code.split("\n# Code:")[1].strip()

    return generated_code

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
code_editor = st.text_area("Write your code:", height=300)
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

# Code Generation
st.header("Code Generation")
code_idea = st.text_input("Enter your code idea:")
if st.button("Generate Code"):
    generated_code = generate_code(code_idea)
    st.code(generated_code, language="python")