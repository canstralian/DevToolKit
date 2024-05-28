import gradio as gr
import os
import subprocess
import time

def chat_with_code(history, user_input):
  """
  Handles user input and processes it through code interpreter and terminal.
  """
  history.append((user_input, None))  # Add user input to history

  try:
    # Attempt to execute code
    if user_input.startswith("```") and user_input.endswith("```"):
      code = user_input[3:-3].strip()
      output = execute_code(code)
    else:
      # Attempt to execute terminal command
      output = execute_terminal(user_input)

    history[-1] = (user_input, output)  # Update history with output
  except Exception as e:
    history[-1] = (user_input, f"Error: {e}")

  return history

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

# Create Gradio interface
iface = gr.ChatInterface(chat_with_code, 
                        title="Code Interpreter & Terminal Chat",
                        description="Ask questions, write code, and run terminal commands!")

iface.launch(share=True)