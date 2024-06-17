import os
import subprocess
import random
from huggingface_hub import InferenceClient
import gradio as gr
from safe_search import safe_search
from i_search import google
from i_search import i_search as i_s
from agent import (
    ACTION_PROMPT,
    ADD_PROMPT,
    COMPRESS_HISTORY_PROMPT,
    LOG_PROMPT,
    LOG_RESPONSE,
    MODIFY_PROMPT,
    PREFIX,
    SEARCH_QUERY,
    READ_PROMPT,
    TASK_PROMPT,
    UNDERSTAND_TEST_RESULTS_PROMPT,
)
from utils import parse_action, parse_file_content, read_python_module_structure
from datetime import datetime
now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)

############################################


VERBOSE = True
MAX_HISTORY = 100
#MODEL = "gpt-3.5-turbo"  # "gpt-4"


def format_prompt(message, history):
  prompt = "<s>"
  for user_prompt, bot_response in history:
    prompt += f"[INST] {user_prompt} [/INST]"
    prompt += f" {bot_response}</s> "
  prompt += f"[INST] {message} [/INST]"
  return prompt



def run_gpt(
    prompt_template,
    stop_tokens,
    max_tokens,
    purpose,
    **prompt_kwargs,
):
    seed = random.randint(1,1111111111111111)
    print (seed)
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
    
    
    #formatted_prompt = format_prompt(f"{system_prompt}, {prompt}", history)
    #formatted_prompt = format_prompt(f'{content}', history)

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
            #response = google(search_return)
            print(response)
            history += "observation: search result is: {}\n".format(response)
        else:
            history += "observation: I need to provide a valid URL to 'action: SEARCH action_input=https://URL'\n"
    except Exception as e:
        history += "observation: {}'\n".format(e)
    return "MAIN", None, history, task

def call_main(purpose, task, history, directory, action_input):
    resp = run_gpt(
        ACTION_PROMPT,
        stop_tokens=["observation:", "task:", "action:","thought:"],
        max_tokens=2096,
        purpose=purpose,
        task=task,
        history=history,
    )
    lines = resp.strip().strip("\n").split("\n")
    for line in lines:
        if line == "":
            continue
        if line.startswith("thought: "):
            history += "{}\n".format(line)
        elif line.startswith("action: "):
            
            action_name, action_input = parse_action(line)
            print (f'ACTION_NAME :: {action_name}')
            print (f'ACTION_INPUT :: {action_input}')
            
            history += "{}\n".format(line)
            if "COMPLETE" in action_name or "COMPLETE" in action_input:
                task = "END"
                return action_name, action_input, history, task
            else:
                return action_name, action_input, history, task
        else:
            history += "{}\n".format(line)
            #history += "observation: the following command did not produce any useful output: '{}', I need to check the commands syntax, or use a different command\n".format(line)
            
            #return action_name, action_input, history, task
            #assert False, "unknown action: {}".format(line)
    return "MAIN", None, history, task


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

NAME_TO_FUNC = {
    "MAIN": call_main,
    "UPDATE-TASK": call_set_task,
    "SEARCH": call_search,
    "COMPLETE": end_fn,

}

def run_action(purpose, task, history, directory, action_name, action_input):
    print(f'action_name::{action_name}')
    try:
        if "RESPONSE" in action_name or "COMPLETE" in action_name:
            action_name="COMPLETE"
            task="END"
            return action_name, "COMPLETE", history, task
    
        # compress the history when it is long
        if len(history.split("\n")) > MAX_HISTORY:
            if VERBOSE:
                print("COMPRESSING HISTORY")
            history = compress_history(purpose, task, history, directory)
        if not action_name in NAME_TO_FUNC:
            action_name="MAIN"
        if action_name == "" or action_name == None:
            action_name="MAIN"
        assert action_name in NAME_TO_FUNC
    
        print("RUN: ", action_name, action_input)
        return NAME_TO_FUNC[action_name](purpose, task, history, directory, action_input)
    except Exception as e:
        history += "observation: the previous command did not produce any useful output, I need to check the commands syntax, or use a different command\n"

        return "MAIN", None, history, task

def run(purpose,history):
    
    #print(purpose)
    #print(hist)
    task=None
    directory="./"
    if history:
        history=str(history).strip("[]")
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
        #yield ("",[(purpose,history)])
        if task == "END":
            return (history)
            #return ("", [(purpose,history)])



################################################

def format_prompt(message, history):
  prompt = "<s>"
  for user_prompt, bot_response in history:
    prompt += f"[INST] {user_prompt} [/INST]"
    prompt += f" {bot_response}</s> "
  prompt += f"[INST] {message} [/INST]"
  return prompt
agents =[
    "WEB_DEV",
    "AI_SYSTEM_PROMPT",
    "PYTHON_CODE_DEV"
]
def generate(
        prompt, history, agent_name=agents[0], sys_prompt="", temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    seed = random.randint(1,1111111111111111)

    agent=prompts.WEB_DEV
    if agent_name == "WEB_DEV":
        agent = prompts.WEB_DEV
    if agent_name == "AI_SYSTEM_PROMPT":
        agent = prompts.AI_SYSTEM_PROMPT
    if agent_name == "PYTHON_CODE_DEV":
        agent = prompts.PYTHON_CODE_DEV        
    system_prompt=agent
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=seed,
    )

    formatted_prompt = format_prompt(f"{system_prompt}, {prompt}", history)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output


additional_inputs=[
    gr.Dropdown(
        label="Agents",
        choices=[s for s in agents],
        value=agents[0],
        interactive=True,
        ),
    gr.Textbox(
        label="System Prompt",
        max_lines=1,
        interactive=True,
    ),
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),

    gr.Slider(
        label="Max new tokens",
        value=1048*10,
        minimum=0,
        maximum=1048*10,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    ),


]

examples=[
    ["Create a basic Python web app using Flask.", None, None, None, None, None, ],
    ["Build a simple Streamlit app to display a data visualization.", None, None, None, None, None, ],
    ["I need a Gradio interface for a machine learning model that takes an image as input and outputs a classification.", None, None, None, None, None, ],
    ["Generate a Python script to scrape data from a website.", None, None, None, None, None, ],
    ["I'm building a React app. How can I use Axios to make API calls?", None, None, None, None, None, ],
    ["Write a Python function to read data from a CSV file.", None, None, None, None, None, ],
    ["I want to deploy my Flask app to Heroku.", None, None, None, None, None, ],
    ["Explain the difference between Git and GitHub.", None, None, None, None, None, ],
    ["How can I use Docker to containerize my Python app?", None, None, None, None, None, ],
    ["I need a simple API endpoint for my web app using Flask.", None, None, None, None, None, ],
    ["Create a function in Python to calculate the factorial of a number.", None, None, None, None, None, ],
]

'''
gr.ChatInterface(
fn=run,
chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
title="Mixtral 46.7B\nMicro-Agent\nInternet Search <br> development test",
examples=examples,
concurrency_limit=20,
with gr.Blocks() as ifacea:
    gr.HTML("""TEST""")
ifacea.launch()
).launch()
with gr.Blocks() as iface:
    #chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    chatbot=gr.Chatbot()
    msg = gr.Textbox()
    with gr.Row():
        submit_b = gr.Button()
        clear = gr.ClearButton([msg, chatbot])
    submit_b.click(run, [msg,chatbot],[msg,chatbot])
    msg.submit(run, [msg, chatbot], [msg, chatbot])
iface.launch()
'''
gr.ChatInterface(
    fn=run,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    title="Mixtral 46.7B\nMicro-Agent\nInternet Search <br> development test",
    examples=examples,
    concurrency_limit=20,
).launch(show_api=False)


Implementation of Next Steps:

Terminal Integration:

Install Libraries: Install either streamlit-terminal or gradio-terminal depending on your chosen framework.
Integrate the Terminal: Use the library's functions to embed a terminal component within your Streamlit or Gradio app.
Capture Input: Capture the user's input from the terminal and pass it to your command execution function.
Display Output: Display the output of the terminal commands, including both standard output and errors.
Code Generation:

LLM Selection: Choose a Hugging Face Transformer model that is suitable for code generation (e.g., google/flan-t5-xl, Salesforce/codet5-base, microsoft/CodeGPT-small).
Prompt Engineering: Develop effective prompts for the LLM to generate code based on natural language instructions.
Code Translation Function: Create a function that takes natural language input, passes it to the LLM with the appropriate prompt, and then returns the generated code.
Code Correction: You can explore ways to automatically correct code errors, perhaps using a combination of syntax checking and LLM assistance.
Workspace Explorer:

Streamlit or Gradio Filesystem Access: Use Streamlit's st.file_uploader or Gradio's gr.File component to allow users to upload files.
File Management: Implement functions to create, edit, and delete files and directories within the workspace.
Display Files: Use Streamlit's st.code or Gradio's gr.File component to display the contents of files in the workspace.
Directory Structure: Display the directory structure of the workspace using a tree-like representation.
Dependency Management:

Package Installation: Create a function that takes a package name as input, installs it using pip, and updates the requirements.txt file.
Workspace Population: Develop a function to create files and directories in the workspace based on installed packages.
Application Build and Launch:

Build Logic: Develop a function to build the web app based on the user's code and dependencies.
Launch Functionality: Implement a mechanism to launch the built app.
Error Correction: Identify and correct errors during the build and launch process.
Automated Assistance: Provide automated assistance during the build and launch process, with a gradient slider to adjust the level of user override.

Recommendations, Enhancements, Optimizations, and Workflow:

1. LLM Selection for Code Generation:
    *  **Google/Flan-T5-XL:** Excellent for code generation, particularly for Python.
    *  **Salesforce/CodeT5-Base:**  Strong for code generation, with a focus on code summarization and translation.
    *  **Microsoft/CodeGPT-Small:**  A smaller model that is suitable for code generation tasks, especially if you have limited computational resources.

2. Prompt Engineering for Code Generation:
    *  **Contextual Prompts:** Provide the LLM with as much context as possible, including the desired programming language, libraries, and any specific requirements.
    *  **Code Snippets:**  If possible, include code snippets as part of the prompt to guide the LLM's code generation.
    *  **Iterative Refinement:**  Use iterative prompting to refine the generated code.  Start with a basic prompt and then provide feedback to the LLM to improve the code.

3. Workspace Exploration:
    *   **Tree-Like View:**  Use a tree-like representation to display the workspace's directory structure.
    *   **Search Functionality:** Implement a search bar to allow users to quickly find specific files or directories.
    *   **Code Highlighting:**  Provide code highlighting for files in the workspace to improve readability.

4. Dependency Management:
    *   **Virtual Environments:** Use virtual environments to isolate project dependencies and prevent conflicts.
    *   **Automatic Updates:**  Implement a mechanism to automatically update dependencies when new versions are available.
    *   **Dependency Locking:**  Use tools like `pip-tools` or `poetry` to lock dependencies to specific versions, ensuring consistent builds.

5. Application Build and Launch:
    *   **Build Tool Integration:**  Consider integrating a build tool like `poetry` or `pipenv` into your workflow to automate the build process.
    *   **Containerization:** Containerize the app using Docker to ensure consistent deployments across different environments.
    *   **Deployment Automation:**  Explore tools like `Heroku`, `AWS Elastic Beanstalk`, or `Google App Engine` to automate the deployment process.

6. Automated Assistance:
    *   **Error Detection and Correction:**  Implement a system that can detect common coding errors and suggest corrections.
    *   **Code Completion:**  Use an LLM to provide code completion suggestions as the user types.