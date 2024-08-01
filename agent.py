import os
import subprocess
import random
from huggingface_hub import InferenceClient
import gradio as gr
from safe_search import safe_search
from i_search import google
from i_search import i_search as i_s
from datetime import datetime
from utils import parse_action, parse_file_content, read_python_module_structure

now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

VERBOSE = True
MAX_HISTORY = 100

# Prompts
ACTION_PROMPT = "action prompt"
ADD_PROMPT = "add prompt"
COMPRESS_HISTORY_PROMPT = "compress history prompt"
LOG_PROMPT = "log prompt"
LOG_RESPONSE = "log response"
MODIFY_PROMPT = "modify prompt"
PREFIX = "prefix"
SEARCH_QUERY = "search query"
READ_PROMPT = "read prompt"
TASK_PROMPT = "task prompt"
UNDERSTAND_TEST_RESULTS_PROMPT = "understand test results prompt"

def format_prompt(message, history):
    prompt = "\n### Instruction:\n{}\n### History:\n{}".format(message, '\n'.join(history))
    return prompt

def run_agent(instruction, history):
    prompt = format_prompt(instruction, history)
    response = ""
    for chunk in generate(prompt, history[-MAX_HISTORY:], temperature=0.7):
        response += chunk
        if "\n\n### Instruction:" in chunk:
            break

    response_actions = []
    for line in response.strip().split('\n'):
        if line.startswith('action:'):
            response_actions.append((line.replace('action: ', '')))

    return response, response_actions

def generate(prompt, history, temperature):
    seed = random.randint(1, 1111111111111111)
    generate_kwargs = {
        "temperature": temperature,
        "max_new_tokens": 256,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "seed": seed,
    }
    formatted_prompt = format_prompt(f"{prompt}", history)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    
    for response in stream:
        output += response.token.text
        yield output

def create_interface():
    global MAX_HISTORY

    block = gr.Blocks()

    chatbot = gr.Chatbot()
    with block.title("Expert Web Developer Assistant"):
        with block.tab("Conversation"):
            txt = gr.Textbox(show_label=False, placeholder="Type something...")
            btn = gr.Button("Send", variant="primary")
            
            txt.submit(run_agent, inputs=[txt, chatbot], outputs=[chatbot, None])
            txt.clear(None, [txt, chatbot]).then(_clear_history, chatbot, _update_chatbot_styles)
            btn.click(_clear_history, chatbot, _update_chatbot_styles)

        with block.tab("Settings"):
            MAX_HISTORY_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Max history", value=MAX_HISTORY)
            MAX_HISTORY_slider.change(lambda x: setattr(block, "MAX_HISTORY", int(x)), MAX_HISTORY_slider)

    return block

def _update_chatbot_styles(history):
    num_messages = sum([1 for item in history if isinstance(item, tuple)])
    gr.Chatbot.update({"num_messages": num_messages})

def _clear_history(history):
    return [], []

# Exportable functions and variables
__all__ = [
    "run_agent",
    "create_interface",
    "format_prompt",
    "generate",
    "MAX_HISTORY",
    "client",
    "VERBOSE",
    "date_time_str",
]