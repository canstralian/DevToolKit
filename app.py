import streamlit as st
import os
import subprocess
import time

st.set_page_config(page_title="Code Interpreter & Terminal Chat", page_icon="🤖")

st.title("Code Interpreter & Terminal Chat")
st.markdown("Ask questions, write code, and run terminal commands!")

chat_history = []

def chat_with_code(user_input):
  """
  Handles user input and processes it through code interpreter and terminal.
  """
  chat_history.append((user_input, None))  # Add user input to history

  try:
    # Attempt to execute code
    if user_input.startswith("```") and user_input.endswith("```"):
      code = user_input[3:-3].strip()
      output = execute_code(code)
    else:
      # Attempt to execute terminal command
      output = execute_terminal(user_input)

    chat_history[-1] = (user_input, output)  # Update history with output
  except Exception as e:
    chat_history[-1] = (user_input, f"Error: {e}")

  return chat_history

def execute_code(code):
  """
  Executes Python code and returns the output.
  """
  try:
    exec(code)
  except Exception as e:
    return f"Error: {e}"

  return "Code executed successfully!"

def execute_terminal(command):
  """
  Executes a terminal command and returns the output.
  """
  process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  output = stdout.decode("utf-8").strip()
  if stderr:
    output += f"\nError: {stderr.decode('utf-8').strip()}"
  return output

# Display chat history
for message in chat_history:
  with st.chat_message(message[0]):
    if message[1]:
      st.write(message[1])

# User input area
user_input = st.text_input("Enter your message or code:", key="input")

# Process user input when Enter is pressed
if user_input:
  chat_history = chat_with_code(user_input)

  # Display updated chat history
  for message in chat_history:
    with st.chat_message(message[0]):
      if message[1]:
        st.write(message[1])

  # Clear the input field
  st.session_state.input = ""