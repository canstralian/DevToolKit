import os
import subprocess
import random
from huggingface_hub import InferenceClient
import gradio as gr
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


def _parse_text_generation_error(error: Optional[str], error_type: Optional[str]) -> TextGenerationError:
    if error_type == "overloaded":
        return OverloadedError(error)  # type: ignore
        
def run_gpt(
    prompt_template,
    stop_tokens,
    max_tokens,
    module_summary,
    purpose,
    **prompt_kwargs,
):
    seed = random.randint(1,1111111111111111)

    generate_kwargs = dict(
        temperature=0.9,
        max_new_tokens=1048,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        seed=seed,
    )

    
    content = PREFIX.format(
        date_time_str=date_time_str,
        purpose=purpose,
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
    module_summary, _, _ = read_python_module_structure(directory)
    resp = run_gpt(
        COMPRESS_HISTORY_PROMPT,
        stop_tokens=["observation:", "task:", "action:", "thought:"],
        max_tokens=512,
        module_summary=module_summary,
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
            history += "observation: I need to provide a valid URL to 'action: SEARCH action_input=URL'\n"
    except Exception as e:
        history += "observation: {}'\n".format(e)
    return "MAIN", None, history, task

def call_main(purpose, task, history, directory, action_input):
    module_summary, _, _ = read_python_module_structure(directory)
    resp = run_gpt(
        ACTION_PROMPT,
        stop_tokens=["observation:", "task:"],
        max_tokens=256,
        module_summary=module_summary,
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


def call_test(purpose, task, history, directory, action_input):
    result = subprocess.run(
        ["python", "-m", "pytest", "--collect-only", directory],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        history += "observation: there are no tests! Test should be written in a test folder under {}\n".format(
            directory
        )
        return "MAIN", None, history, task
    result = subprocess.run(
        ["python", "-m", "pytest", directory], capture_output=True, text=True
    )
    if result.returncode == 0:
        history += "observation: tests pass\n"
        return "MAIN", None, history, task
    module_summary, content, _ = read_python_module_structure(directory)
    resp = run_gpt(
        UNDERSTAND_TEST_RESULTS_PROMPT,
        stop_tokens=[],
        max_tokens=256,
        module_summary=module_summary,
        purpose=purpose,
        task=task,
        history=history,
        stdout=result.stdout[:5000],  # limit amount of text
        stderr=result.stderr[:5000],  # limit amount of text
    )
    history += "observation: tests failed: {}\n".format(resp)
    return "MAIN", None, history, task


def call_set_task(purpose, task, history, directory, action_input):
    module_summary, content, _ = read_python_module_structure(directory)
    task = run_gpt(
        TASK_PROMPT,
        stop_tokens=[],
        max_tokens=64,
        module_summary=module_summary,
        purpose=purpose,
        task=task,
        history=history,
    ).strip("\n")
    history += "observation: task has been updated to: {}\n".format(task)
    return "MAIN", None, history, task


def call_read(purpose, task, history, directory, action_input):
    if not os.path.exists(action_input):
        history += "observation: file does not exist\n"
        return "MAIN", None, history, task
    module_summary, content, _ = read_python_module_structure(directory)
    f_content = (
        content[action_input] if content[action_input] else "< document is empty >"
    )
    resp = run_gpt(
        READ_PROMPT,
        stop_tokens=[],
        max_tokens=256,
        module_summary=module_summary,
        purpose=purpose,
        task=task,
        history=history,
        file_path=action_input,
        file_contents=f_content,
    ).strip("\n")
    history += "observation: {}\n".format(resp)
    return "MAIN", None, history, task


def call_modify(purpose, task, history, directory, action_input):
    if not os.path.exists(action_input):
        history += "observation: file does not exist\n"
        return "MAIN", None, history, task
    (
        module_summary,
        content,
        _,
    ) = read_python_module_structure(directory)
    f_content = (
        content[action_input] if content[action_input] else "< document is empty >"
    )
    resp = run_gpt(
        MODIFY_PROMPT,
        stop_tokens=["action:", "thought:", "observation:"],
        max_tokens=2048,
        module_summary=module_summary,
        purpose=purpose,
        task=task,
        history=history,
        file_path=action_input,
        file_contents=f_content,
    )
    new_contents, description = parse_file_content(resp)
    if new_contents is None:
        history += "observation: failed to modify file\n"
        return "MAIN", None, history, task

    with open(action_input, "w") as f:
        f.write(new_contents)

    history += "observation: file successfully modified\n"
    history += "observation: {}\n".format(description)
    return "MAIN", None, history, task


def call_add(purpose, task, history, directory, action_input):
    d = os.path.dirname(action_input)
    if not d.startswith(directory):
        history += "observation: files must be under directory {}\n".format(directory)
    elif not action_input.endswith(".py"):
        history += "observation: can only write .py files\n"
    else:
        if d and not os.path.exists(d):
            os.makedirs(d)
        if not os.path.exists(action_input):
            module_summary, _, _ = read_python_module_structure(directory)
            resp = run_gpt(
                ADD_PROMPT,
                stop_tokens=["action:", "thought:", "observation:"],
                max_tokens=2048,
                module_summary=module_summary,
                purpose=purpose,
                task=task,
                history=history,
                file_path=action_input,
            )
            new_contents, description = parse_file_content(resp)
            if new_contents is None:
                history += "observation: failed to write file\n"
                return "MAIN", None, history, task

            with open(action_input, "w") as f:
                f.write(new_contents)

            history += "observation: file successfully written\n"
            history += "obsertation: {}\n".format(description)
        else:
            history += "observation: file already exists\n"
    return "MAIN", None, history, task
def end_fn(purpose, task, history, directory, action_input):
    task = "END"
    return "COMPLETE", None, history, task
NAME_TO_FUNC = {
    "MAIN": call_main,
    "UPDATE-TASK": call_set_task,
    "SEARCH": call_search,
    "COMPLETE": end_fn,

}


def run_action(purpose, task, history, directory, action_name, action_input):
    if "RESPONSE" in action_name:
        task="END"
        return action_name, action_input, history, task

    # compress the history when it is long
    if len(history.split("\n")) > MAX_HISTORY:
        if VERBOSE:
            print("COMPRESSING HISTORY")
        history = compress_history(purpose, task, history, directory)

    assert action_name in NAME_TO_FUNC

    print("RUN: ", action_name, action_input)
    return NAME_TO_FUNC[action_name](purpose, task, history, directory, action_input)


def run(purpose,hist):
    
    print(purpose)
    print(hist)
    task=None
    directory="./"
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
        if task == "END":
            return history



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

examples=[["I'm planning a vacation to Japan. Can you suggest a one-week itinerary including must-visit places and local cuisines to try?", None, None, None, None, None, ],
          ["Can you write a short story about a time-traveling detective who solves historical mysteries?", None, None, None, None, None,],
          ["I'm trying to learn French. Can you provide some common phrases that would be useful for a beginner, along with their pronunciations?", None, None, None, None, None,],
          ["I have chicken, rice, and bell peppers in my kitchen. Can you suggest an easy recipe I can make with these ingredients?", None, None, None, None, None,],
          ["Can you explain how the QuickSort algorithm works and provide a Python implementation?", None, None, None, None, None,],
          ["What are some unique features of Rust that make it stand out compared to other systems programming languages like C++?", None, None, None, None, None,],
         ]


gr.ChatInterface(
    fn=run,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    title="Mixtral 46.7B\nMicro-Agent\nInternet Search",
    examples=examples,
    concurrency_limit=20,
).launch(show_api=False)