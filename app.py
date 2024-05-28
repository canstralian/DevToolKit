import streamlit as st
import os
import subprocess
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import black
from pylint import epylint as lint

PROJECT_ROOT = "projects"

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
    model_name = "gpt2"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Truncate input text to avoid exceeding the model's maximum length
    max_input_length = 900
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    if input_ids.shape[1] > max_input_length:
        input_ids = input_ids[:, :max_input_length]

    # Generate chatbot response
    outputs = model.generate(
        input_ids, max_new_tokens=50, num_return_sequences=1, do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# 2. Terminal
def terminal_interface(command, project_name=None):
    """Executes commands in the terminal.

    Args:
        command: User's command.
        project_name: Name of the project workspace to add installed packages.

    Returns:
        The terminal output.
    """
    # Execute command
    try:
        process = subprocess.run(command.split(), capture_output=True, text=True)
        output = process.stdout

        # If the command is to install a package, update the workspace
        if "install" in command and project_name:
            requirements_path = os.path.join(PROJECT_ROOT, project_name, "requirements.txt")
            with open(requirements_path, "a") as req_file:
                package_name = command.split()[-1]
                req_file.write(f"{package_name}\n")
    except Exception as e:
        output = f"Error: {e}"
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
        (pylint_stdout, pylint_stderr) = lint.py_run(code, return_std=True)
        lint_message = pylint_stdout.getvalue()
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
    project_path = os.path.join(PROJECT_ROOT, project_name)
    # Create project directory
    try:
        os.makedirs(project_path)
        requirements_path = os.path.join(project_path, "requirements.txt")
        with open(requirements_path, "w") as req_file:
            req_file.write("")  # Initialize an empty requirements.txt file
        status = f'Project "{project_name}" created successfully.'
    except FileExistsError:
        status = f'Project "{project_name}" already exists.'
    return status

def add_code_to_workspace(project_name, code, file_name):
    """Adds selected code files to the workspace.

    Args:
        project_name: Name of the project.
        code: Code to be added.
        file_name: Name of the file to be created.

    Returns:
        File creation status.
    """
    project_path = os.path.join(PROJECT_ROOT, project_name)
    file_path = os.path.join(project_path, file_name)

    try:
        with open(file_path, "w") as code_file:
            code_file.write(code)
        status = f'File "{file_name}" added to project "{project_name}" successfully.'
    except Exception as e:
        status = f"Error: {e}"
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
    # Load the summarization model
    model_name = "facebook/bart-large-cnn"
    try:
        summarizer = pipeline("summarization", model=model_name)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Truncate input text to avoid exceeding the model's maximum length
    max_input_length = 1024
    inputs = text
    if len(text) > max_input_length:
        inputs = text[:max_input_length]

    # Generate summary
    summary = summarizer(inputs, max_length=100, min_length=30, do_sample=False)[0][
        "summary_text"
    ]
    return summary

# Example: Sentiment analysis tool
def sentiment_analysis(text):
    """Performs sentiment analysis on a given text using a Hugging Face model.

    Args:
        text: Text to be analyzed.

    Returns:
        Sentiment analysis result.
    """
    # Load the sentiment analysis model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    try:
        analyzer = pipeline("sentiment-analysis", model=model_name)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Perform sentiment analysis
    result = analyzer(text)[0]
    return result

# Example: Text translation tool
def translate_text(text, target_language="fr"):
    """Translates a given text to the target language using a Hugging Face model.

    Args:
        text: Text to be translated.
        target_language: The language to translate the text to.

    Returns:
        Translated text.
    """
    # Load the translation model
    model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
    try:
        translator = pipeline("translation", model=model_name)
    except EnvironmentError as e:
        return f"Error loading model: {e}"

    # Translate text
    translated_text = translator(text)[0]["translation_text"]
    return translated_text


# 6. Code Generation
def generate_code(idea):
    """Generates code based on a given idea using the EleutherAI/gpt-neo-2.7B model.

    Args:
        idea: The idea for the code to be generated.

    Returns:
        The generated code as a string.
    """

    # Load the code generation model
    model_name = "EleutherAI/gpt-neo-2.7B"
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

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Tool Box", "Workspace Chat App"])

if app_mode == "Tool Box":
    # Tool Box
    st.header("AI-Powered Tools")

    # Chat Interface
    st.subheader("Chat with CodeCraft")
    chat_input = st.text_area("Enter your message:")
    if st.button("Send"):
        chat_response = chat_interface(chat_input)
        st.write(f"CodeCraft: {chat_response}")

    # Terminal Interface
    st.subheader("Terminal")
    terminal_input = st.text_input("Enter a command:")
    if st.button("Run"):
        terminal_output = terminal_interface(terminal_input)
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

    # Text Translation Tool
    st.subheader("Translate Text")
    translation_text = st.text_area("Enter text to translate:")
    target_language = st.text_input("Enter target language code (e.g., 'fr' for French):")
    if st.button("Translate"):
        translated_text = translate_text(translation_text, target_language)
        st.write(f"Translated Text: {translated_text}")

    # Code Generation
    st.subheader("Code Generation")
    code_idea = st.text_input("Enter your code idea:")
    if st.button("Generate Code"):
        generated_code = generate_code(code_idea)
        st.code(generated_code, language="python")

elif app_mode == "Workspace Chat App":
    # Workspace Chat App
    st.header("Workspace Chat App")

    # Project Workspace Creation
    st.subheader("Create a New Project")
    project_name = st.text_input("Enter project name:")
    if st.button("Create Project"):
        workspace_status = workspace_interface(project_name)
        st.success(workspace_status)

    # Add Code to Workspace
    st.subheader("Add Code to Workspace")
    code_to_add = st.text_area("Enter code to add to workspace:")
    file_name = st.text_input("Enter file name (e.g., 'app.py'):")
    if st.button("Add Code"):
        add_code_status = add_code_to_workspace(project_name, code_to_add, file_name)
        st.success(add_code_status)

    # Terminal Interface with Project Context
    st.subheader("Terminal (Workspace Context)")
    terminal_input = st.text_input("Enter a command within the workspace:")
    if st.button("Run Command"):
        terminal_output = terminal_interface(terminal_input, project_name)
        st.code(terminal_output, language="bash")

    # Chat Interface for Guidance
    st.subheader("Chat with CodeCraft for Guidance")
    chat_input = st.text_area("Enter your message for guidance:")
    if st.button("Get Guidance"):
        chat_response = chat_interface(chat_input)
        st.write(f"CodeCraft: {chat_response}")