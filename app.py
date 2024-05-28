import streamlit as st
import os
import subprocess
import random
import string
from huggingface_hub import cached_download, hf_hub_url
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import black
import pylint
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

# 6. Code Generation
def generate_code(idea):
    """Generates code based on a given idea using the bigscience/T0_3B model.

    Args:
        idea: The idea for the code to be generated.

    Returns:
        The generated code as a string.
    """

    # Load the code generation model
    model_name = 'bigscience/T0_3B'  # Choose your model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

# 7. Sentiment Analysis
def analyze_sentiment(text):
    """Analyzes the sentiment of a given text.

    Args:
        text: The text to analyze.

    Returns:
        A dictionary containing the sentiment label and score.
    """
    model_name = 'distilbert-base-uncased-finetuned-sst-3-literal-labels'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    result = classifier(text)[0]
    return result

# 8. Text Translation
def translate_text(text, target_language):
    """Translates a given text to the specified target language.

    Args:
        text: The text to translate.
        target_language: The target language code (e.g., 'fr' for French, 'es' for Spanish).

    Returns:
        The translated text.
    """
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")  # Example: English to Spanish
    translation = translator(text, target_lang=target_language)[0]['translation_text']
    return translation

# Streamlit App
st.title("CodeCraft: Your AI-Powered Development Toolkit")

# Workspace Selection
st.sidebar.header("Select Workspace")
project_name = st.sidebar.selectbox("Choose a project", os.listdir('projects'))

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
code_editor = st.text_area("Write your code:", language="python", height=300)
if st.button("Format & Lint"):
    formatted_code, lint_message = code_editor_interface(code_editor)
    st.code(formatted_code, language="python")
    st.info(lint_message)

# AI-Infused Tools
st.header("AI-Powered Tools")

# Text Summarization
st.subheader("Text Summarization")
text_to_summarize = st.text_area("Enter text to summarize:")
if st.button("Summarize"):
    summary = summarize_text(text_to_summarize)
    st.write(f"Summary: {summary}")

# Sentiment Analysis
st.subheader("Sentiment Analysis")
text_to_analyze = st.text_area("Enter text to analyze sentiment:")
if st.button("Analyze Sentiment"):
    sentiment_result = analyze_sentiment(text_to_analyze)
    st.write(f"Sentiment: {sentiment_result['label']}, Score: {sentiment_result['score']}")

# Text Translation
st.subheader("Text Translation")
text_to_translate = st.text_area("Enter text to translate:")
target_language = st.selectbox("Choose target language", ['fr', 'es', 'de', 'zh-CN'])  # Example languages
if st.button("Translate"):
    translation = translate_text(text_to_translate, target_language)
    st.write(f"Translation: {translation}")

# Code Generation
st.header("Code Generation")
code_idea = st.text_input("Enter your code idea:")
if st.button("Generate Code"):
    try:
        generated_code = generate_code(code_idea)
        st.code(generated_code, language="python")
    except Exception as e:
        st.error(f"Error generating code: {e}")

# Launch Chat App (with Authentication)
if st.button("Launch Chat App"):
    # Get the current working directory
    cwd = os.getcwd()

    # User Authentication
hf_token = st.text_input("Enter your Hugging Face Token:")
if hf_token:
    # Set the token using HfFolder
    HfFolder.save_token(hf_token)

    # Construct the command to launch the chat app
    command = f"cd projects/{project_name} && streamlit run chat_app.py"

    # Execute the command
    try:
        process = subprocess.run(command.split(), capture_output=True, text=True)
        st.write(f"Chat app launched successfully!")
    except Exception as e:
        st.error(f"Error launching chat app: {e}")