import os
import subprocess
import random
from huggingface_hub import InferenceClient
import gradio as gr
from safe_search import safe_search
from i_search import google
from i_search import i_search as i_s
from agent import ( run_agent, create_interface, format_prompt_var, generate, MAX_HISTORY, client, VERBOSE, date_time_str, )

from utils import parse_action, parse_file_content, read_python_module_structure
from datetime import datetime

now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

VERBOSE = True
MAX_HISTORY = 100

def format_prompt_var(message, history):
    prompt = " "
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/usr]\n{bot_response}\n"
    prompt += f"[INST] {message} [/usr]\n"
    return prompt

def run_gpt(prompt_template, stop_tokens, max_tokens, purpose, **prompt_kwargs):
    seed = random.randint(1, 1111111111111111)
    print(seed)
    generate_kwargs = dict(
        temperature=1.0,
        max_new_tokens=2096,
        top_p=0.99,
        repetition_penalty=1.0,
        do_sample=True,
        seed=seed,
    )

    content = PREFIX.format(
        date_time_str=date_time_str,
        purpose=purpose,
        safe_search=safe_search,
    ) + prompt_template.format(**prompt_kwargs)
    if VERBOSE:
        print(LOG_PROMPT.format(content))

    stream = client.text_generation(content, **generate_kwargs, stream=True, details=True, return_full_text=False)
    resp = ""
    for response in stream:
        resp += response.token.text

    if VERBOSE:
        print(LOG_RESPONSE.format(resp))
    return resp
def compress_history(purpose, task, history, directory):
    resp = run_gpt(
        COMPRESS_HISTORY_PROMPT,
        stop_tokens=["observation:", "task:", "action:", "thought:"],
        max_tokens=512,
        purpose=purpose,
        task=task,
        history=history,
    )
    history = "observation: {}\n".format(resp)
    return history

def call_search(purpose, task, history, directory, action_input):
    print("CALLING SEARCH")
    try:
        if "http" in action_input:
            if "<" in action_input:
                action_input = action_input.strip("<")
            if ">" in action_input:
                action_input = action_input.strip(">")
            response = i_s(action_input)
            print(response)
            history += "observation: search result is: {}\n".format(response)
        else:
            history += "observation: I need to provide a valid URL to 'action: SEARCH action_input=https://URL'\n"
    except Exception as e:
        history += "{}\n".format(e)  # Fixing this line to include the exception message
    if "COMPLETE" in action_name or "COMPLETE" in action_input:
        task = "END"
    return action_name, action_input, history, task
def call_set_task(purpose, task, history, directory, action_input):
    task = run_gpt(
        TASK_PROMPT,
        stop_tokens=[],
        max_tokens=64,
        purpose=purpose,
        task=task,
        history=history,
    ).strip("\n")
    history += "observation: task has been updated to: {}\n".format(task)
    return "MAIN", None, history, task

def end_fn(purpose, task, history, directory, action_input):
    task = "END"
    return "COMPLETE", "COMPLETE", history, task

EXAMPLE_PROJECT_DIRECTORY = './example_project/'

PREFIX = """Answer the following question as accurately as possible, providing detailed responses that cover each aspect of the topic. Make sure to maintain a professional tone throughout your answers. Also please make sure to meet the safety criteria specified earlier. Question: What are the suggested approaches for creating a responsive navigation bar? Answer:"""
LOG_PROMPT = "Prompt: {}"
LOG_RESPONSE = "Response: {}"
COMPRESS_HISTORY_PROMPT = """Given the context history, compress it down to something meaningful yet short enough to fit into a single chat message without exceeding over 512 tokens. Context: {}"""
TASK_PROMPT = """Determine the correct next step in terms of actions, thoughts or observations for the following task: {}, current history: {}, current directory: {}."""

NAME_TO_FUNC = {
    "MAIN": call_main,
    "UPDATE-TASK": call_set_task,
    "SEARCH": call_search,
    "COMPLETE": end_fn,
}

def _clean_up():
    if os.path.exists(EXAMPLE_PROJECT_DIRECTORY):
        shutil.rmtree(EXAMPLE_PROJECT_DIRECTORY)

def call_main(purpose, task, history, directory, action_input=''):
    _clean_up()
    os.makedirs(EXAMPLE_PROJECT_DIRECTORY)
    template = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
  <style>
    {{%style}}
  </style>
</head>
<body>
  {{%body}}
</body>
</html>'''

    navbar = f'''<nav>
  <input type="checkbox" id="check">
  <label for="check" class="checkbtn">
    <i class="fas fa-bars"></i>
  </label>
  <label class="logo">LOGO</label>
  <ul>
    <li><a href="#home">Home</a></li>
    <li><a href="#about">About Us</a></li>
    <li><a href="#services">Services</a></li>
    <li><a href="#contact">Contact Us</a></li>
  </ul>
</nav>'''

    css = '''*{
  box-sizing: border-box;}

body {{
  font-family: sans-serif;
  margin: 0;
  padding: 0;
  background: #f4f4f4;
}}

/* Navigation */
nav {{
  position: fixed;
  width: 100%;
  height: 70px;
  line-height: 70px;
  z-index: 999;
  transition: all .6s ease-in-out;
}}

nav ul {{
  float: right;
  margin-right: 40px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  list-style: none;
}}

nav li {{
  position: relative;
  text-transform: uppercase;
  letter-spacing: 2px;
  cursor: pointer;
  padding: 0 10px;
}}

nav li:hover > ul {{
  visibility: visible;
  opacity: 1;
  transform: translateY(0);
  top: auto;
  left:auto;
  -webkit-transition:all 0.3s linear; /* Safari/Chrome/Opera/Gecko */
    -moz-transition:all 0.3s linear; /* FF3.6+ */
     -ms-transition:all 0.3s linear; /* IE10 */
      -o-transition:all 0.3s linear; /* Opera 10.5–12.00 */
          transition:all 0.3s linear;
}}

nav ul ul {{
  visibility: hidden;
  opacity: 0;
  min-width: 180px;
  white-space: nowrap;
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0px 0px 3px rgba(0, 0, 0, 0.2);
  border-radius: 0px;
  transition: all 0.5s cubic-bezier(0.770, 0.000, 0.175, 1.000);
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 9999;
  padding: 0;
}}'''

    with open(os.path.join(EXAMPLE_PROJECT_DIRECTORY, 'index.html'), 'w') as f:
        f.write(template.format(body=navbar, style=css))

    return "MAIN", "", f"Created a responsive navigation bar in:\n{EXAMPLE_PROJECT_DIRECTORY}", task

def run_action(purpose, task, history, directory, action_name, action_input):
    print(f'action_name::{action_name}')
    try:
        if "RESPONSE" in action_name or "COMPLETE" in action_name:
            action_name = "COMPLETE"
            task = "END"
            return action_name, "COMPLETE", history, task

        if len(history.split('\n')) > MAX_HISTORY:
            if VERBOSE:
                print("COMPRESSING HISTORY")
            history = compress_history(purpose, task, history, directory)
        if not action_name in NAME_TO_FUNC:
            action_name = "MAIN"
        if action_name == '' or action_name is None:
            action_name = "MAIN"
        assert action_name in NAME_TO_FUNC

        print("RUN: ", action_name, action_input)
        return NAME_TO_FUNC[action_name](purpose, task, history, directory, action_input)
    except Exception as e:
        history += "observation: the previous command did not produce any useful output, I need to check the commands syntax, or use a different command\n"
        return "MAIN", None, history, task
def run(purpose, history):
    task = None
    directory = "./"
    if history:
        history = str(history).strip("[]")
        if not history:
            history = ""

    action_name = "UPDATE-TASK" if task is None else "MAIN"
    action_input = None
    while True:
        print("")
        print("")
        print("---")
        print("purpose:", purpose)
        print("task:", task)
        print("---")
        print(history)
        print("---")

        action_name, action_input, history, task = run_action(
            purpose,
            task,
            history,
            directory,
            action_name,
            action_input,
        )
        yield (history)
        if task == "END":
            return (history)
    iface = gr.Interface(fn=run, inputs=["text", "text"], outputs="text", title="Expert Web Developer Assistant Agent", description="Ask me questions, give me tasks, and I will respond accordingly.\n Example: 'Purpose: Create a contact form | Action: FORMAT INPUT' & Input: '<form><div><label for='email'>Email:</label><input type='email'/></div></form>' ")

    
    # Launch the Gradio interface
    iface.launch(share=True)
  
if __name__ == "__main__":
    main("Sample Purpose", "Sample History")