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
        history += "{}\n".format(line)
            if "COMPLETE" in action_name or "COMPLETE" in action_input:
                task = "END"
                return action_name, action_input, history, task
        else:
            history += "{}\n".format(line)
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
            action_name = "COMPLETE"
            task = "END"
            return action_name, "COMPLETE", history, task

        if len(history.split("\n")) > MAX_HISTORY:
            if VERBOSE:
                print("COMPRESSING HISTORY")
            history = compress_history(purpose, task, history, directory)
        if not action_name in NAME_TO_FUNC:
            action_name = "MAIN"
        if action_name == "" or action_name is None:
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
    iface = gr.Interface(fn=run, inputs=["text", "text"], outputs="text", title="Interactive AI Assistant", description="Enter your purpose and history to interact with the AI assistant.")
    
    # Launch the Gradio interface
    iface.launch(share=True)
  
if __name__ == "__main__":
    main("Sample Purpose", "Sample History")