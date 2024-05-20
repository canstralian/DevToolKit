import gradio as gr
from transformers import pipeline

# Assuming you have access to OpenAI's API and Codex
# Replace 'your_api_key' with your actual OpenAI API key
code_completion = pipeline("text-generation", model="code-davinci-002", temperature=0.7, max_length=50, num_return_sequences=1, api_key='your_api_key')

def generate_code(input_code):
  return code_completion(input_code, max_length=50, num_return_sequences=1)[0]['generated_text']

iface = gr.Interface(
  fn=generate_code,
  inputs=gr.inputs.Textbox(label="Enter your code snippet"),
  outputs=gr.outputs.Textbox(label="Generated Code"),
  title="Code Completion Assistant"
)

iface.launch()