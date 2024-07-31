import os
import subprocess

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from agent.prompts import (
    ACTION_PROMPT,
    ADD_PROMPT,
    COMPRESS_HISTORY_PROMPT,
    LOG_PROMPT,
    LOG_RESPONSE,
    MODIFY_PROMPT,
    PREFIX,
    READ_PROMPT,
    TASK_PROMPT,
    UNDERSTAND_TEST_RESULTS_PROMPT,
)
from agent.utils import parse_action, parse_file_content, read_python_module_structure

# Hugging Face model and tokenizer setup
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

VERBOSE = False
MAX_HISTORY = 100

def run_gpt(prompt_template, stop_tokens, max_tokens, module_summary, purpose, **prompt_kwargs):
    content = PREFIX.format(
        module_summary=module_summary,
        purpose=purpose,
    ) + prompt_template.format(**prompt_kwargs)
    if VERBOSE:
        st.write(LOG_PROMPT.format(content))
    resp = generator(content, max_length=max_tokens, stop=stop_tokens)[0]["generated_text"]
    if VERBOSE:
        st.write(LOG_RESPONSE.format(resp))
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
            history += "{}\n".format(line)
            return action_name, action_input, history, task
        else:
            assert False, "unknown action: {}".format(line)
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
            history += "observation: {}\n".format(description)
        else:
            history += "observation: file already exists\n"
    return "MAIN", None, history, task

# Streamlit UI
st.title("AI Powered Code Assistant")

with st.sidebar:
    st.header("Task Configuration")
    purpose = st.text_input("Purpose")
    task = st.text_input("Task")
    directory = st.text_input("Directory")
    action_input = st.text_input("Action Input")
    action = st.selectbox("Action", ["main", "test", "set_task", "read", "modify", "add"])

if st.button("Run Action"):
    history = ""
    if action == "main":
        action_name, action_input, history, task = call_main(purpose, task, history, directory, action_input)
    elif action == "test":
        action_name, action_input, history, task = call_test(purpose, task, history, directory, action_input)
    elif action == "set_task":
        action_name, action_input, history, task = call_set_task(purpose, task, history, directory, action_input)
    elif action == "read":
        action_name, action_input, history, task = call_read(purpose, task, history, directory, action_input)
    elif action == "modify":
        action_name, action_input, history, task = call_modify(purpose, task, history, directory, action_input)
    elif action == "add":
        action_name, action_input, history, task = call_add(purpose, task, history, directory, action_input)
    
    st.subheader("History")
    st.write(history)