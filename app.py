import streamlit as st
import os
import subprocess
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import black
from pylint import lint
from io import StringIO
import openai
import sys

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

PROJECT_ROOT = "projects"

# Global state to manage communication between Tool Box and Workspace Chat App
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'terminal_history' not in st.session_state:
    st.session_state.terminal_history = []
if 'workspace_projects' not in st.session_state:
    st.session_state.workspace_projects = {}

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
        st.session_state.workspace_projects[project_name] = {'files': []}
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
        st.session_state.workspace_projects[project_name]['files'].append(file_name)
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

# Example: Text translation tool (code translation)
def translate_code(code, source_language, target_language):
    """Translates code from one programming language to another using OpenAI Codex.

    Args:
        code: Code to be translated.
        source_language: The source programming language.
        target_language: The target programming language.

    Returns:
        Translated code.
    """
    prompt = f"Translate the following {source_language} code to {target_language}:\n\n{code}"
    try:
        response = openai.Completion.create(
            engine="code-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            temperature=0.3,
            top_p=1,
            n=1,
            stop=None
        )
        translated_code = response.choices[0].text.strip()
    except Exception as e:
        translated_code = f"Error: {e}"
    return translated_code


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
    parts = generated_code.split("\n# Code:")
    if len(parts) > 1:
        generated_code = parts[1].strip()
    else:
        generated_code = generated_code.strip()

    return generated_code


# 7. AI Personas Creator
def create_persona_from_text(text):
    """Creates an AI persona from the given text.

    Args:
        text: Text to be used for creating the persona.

    Returns:
        Persona prompt.
    """
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


# Streamlit App
st.title("AI Personas Creator")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["AI Personas Creator", "Tool Box", "Workspace Chat App"])

if app_mode == "AI Personas Creator":
    # AI Personas Creator
    st.header("Create the System Prompt of an AI Persona from YouTube or Text")

    st.subheader("From Text")
    text_input = st.text_area("Enter text to create an AI persona:")
    if st.button("Create Persona"):
        persona_prompt = create_persona_from_text(text_input)
        st.subheader("Persona Prompt")
        st.text_area("You may now copy the text below and use it as Custom prompt!", value=persona_prompt, height=300)

elif app_mode == "Tool Box":
    # Tool Box
    st.header("AI-Powered Tools")

    # Chat Interface
    st.subheader("Chat with CodeCraft")
    chat_input = st.text_area("Enter your message:")
    if st.button("Send"):
        chat_response = chat_interface(chat_input)
        st.session_state.chat_history.append((chat_input, chat_response))
        st.write(f"CodeCraft: {chat_response}")

    # Terminal Interface
    st.subheader("Terminal")
    terminal_input = st.text_input("Enter a command:")
    if st.button("Run"):
        terminal_output = terminal_interface(terminal_input)
        st.session_state.terminal_history.append((terminal_input, terminal_output))
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

    # Text Translation Tool (Code Translation)
    st.subheader("Translate Code")
    code_to_translate = st.text_area("Enter code to translate:")
    source_language = st.text_input("Enter source language (e.g., 'Python'):")
    target_language = st.text_input("Enter target language (e.g., 'JavaScript'):")
    if st.button("Translate Code"):
        translated_code = translate_code(code_to_translate, source_language, target_language)
        st.code(translated_code, language=target_language.lower())

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
        st.session_state.chat_history.append((chat_input, chat_response))
        st.write(f"CodeCraft: {chat_response}")

    # Display Chat History
    st.subheader("Chat History")
    for user_input, response in st.session_state.chat_history:
        st.write(f"User: {user_input}")
        st.write(f"CodeCraft: {response}")

    # Display Terminal History
    st.subheader("Terminal History")
    for command, output in st.session_state.terminal_history:
        st.write(f"Command: {command}")
        st.code(output, language="bash")

    # Display Projects and Files
    st.subheader("Workspace Projects")
    for project, details in st.session_state.workspace_projects.items():
        st.write(f"Project: {project}")
        for file in details['files']:
            st.write(f"  - {file}")