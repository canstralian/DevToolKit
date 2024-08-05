import os
import subprocess
import time
from typing import List, Dict

from huggingface_hub import InferenceClient
import streamlit as st

from .prompts import * (
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
    WEB_DEV_SYSTEM_PROMPT,
    AI_SYSTEM_PROMPT,
    WEB_DEV,
    PYTHON_CODE_DEV,
    HUGGINGFACE_FILE_DEV,
)
from app.utils import (
    parse_action,
    parse_file_content,
    read_python_module_structure,
    extract_imports,  # Unused import, consider removing or using
    get_file,  # Unused import, consider removing or using
)

# --- Constants ---
AGENT_TYPES = [
    "Task Executor",
    "Information Retriever",
    "Decision Maker",
    "Data Analyzer",
]
TOOL_TYPES = [
    "Web Scraper",
    "Database Connector",
    "API Caller",
    "File Handler",
    "Text Processor",
]
VERBOSE = False
MAX_HISTORY = 100
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Consider using a smaller model

# --- Initialize Hugging Face client ---
client = InferenceClient(MODEL)


# --- Classes ---
class Agent:
    def __init__(self, name: str, agent_type: str, complexity: int):
        self.name = name
        self.type = agent_type
        self.complexity = complexity
        self.tools: List[Tool] = []

    def add_tool(self, tool: "Tool"):
        self.tools.append(tool)

    def __str__(self):
        return f"{self.name} ({self.type}) - Complexity: {self.complexity}"


class Tool:
    def __init__(self, name: str, tool_type: str):
        self.name = name
        self.type = tool_type

    def __str__(self):
        return f"{self.name} ({self.type})"


class Pypelyne:
    def __init__(self):
        self.agents: List[Agent] = []
        self.tools: List[Tool] = []
        self.history: str = ""
        self.task: str = ""
        self.purpose: str = ""
        self.directory: str = ""

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def add_tool(self, tool: Tool):
        self.tools.append(tool)

    def generate_chat_app(self) -> str:
        time.sleep(2)  # Simulate processing time
        return f"Chat app generated with {len(self.agents)} agents and {len(self.tools)} tools."

    def run_gpt(
        self, prompt_template: str, stop_tokens: List[str], max_tokens: int, **prompt_kwargs
    ) -> str:
        content = (
            PREFIX.format(
                module_summary=read_python_module_structure(self.directory)[0],
                purpose=self.purpose,
            )
            + prompt_template.format(**prompt_kwargs)
        )

        if VERBOSE:
            print(LOG_PROMPT.format(content))

        try:
            stream = client.text_generation(
                prompt=content,
                max_new_tokens=max_tokens,
                stop_sequences=stop_tokens if stop_tokens else None,
                do_sample=True,
                temperature=0.7,
            )
            resp = "".join(token for token in stream)
        except Exception as e:
            print(f"Error in run_gpt: {e}")
            resp = f"Error: {e}"

        if VERBOSE:
            print(LOG_RESPONSE.format(resp))
        return resp

    def compress_history(self):
        resp = self.run_gpt(
            COMPRESS_HISTORY_PROMPT,
            stop_tokens=["observation:", "task:", "action:", "thought:"],
            max_tokens=512,
            task=self.task,
            history=self.history,
        )
        self.history = f"observation: {resp}\n"

    def run_action(self, action_name: str, action_input: str) -> str:
        if action_name == "COMPLETE":
            return "Task completed."

        if len(self.history.split("\n")) > MAX_HISTORY:
            if VERBOSE:
                print("COMPRESSING HISTORY")
            self.compress_history()

        action_funcs = {
            "MAIN": self.call_main,
            "UPDATE-TASK": self.call_set_task,
            "MODIFY-FILE": self.call_modify,
            "READ-FILE": self.call_read,
            "ADD-FILE": self.call_add,
            "TEST": self.call_test,
        }

        if action_name not in action_funcs:
            return f"Unknown action: {action_name}"

        print(f"RUN: {action_name} {action_input}")
        return action_funcs[action_name](action_input)

    def call_main(self, action_input: str) -> str:
        resp = self.run_gpt(
            ACTION_PROMPT,
            stop_tokens=["observation:", "task:"],
            max_tokens=256,
            task=self.task,
            history=self.history,
        )
        lines = resp.strip().strip("\n").split("\n")
        for line in lines:
            if line == "":
                continue
            if line.startswith("thought: "):
                self.history += f"{line}\n"
            elif line.startswith("action: "):
                action_name, action_input = parse_action(line)
                self.history += f"{line}\n"
                return self.run_action(action_name, action_input)
        return "No valid action found."

    def call_set_task(self, action_input: str) -> str:
        self.task = (
            self.run_gpt(
                TASK_PROMPT,
                stop_tokens=[],
                max_tokens=64,
                task=self.task,
                history=self.history,
            )
            .strip("\n")
            .strip()
        )
        self.history += f"observation: task has been updated to: {self.task}\n"
        return f"Task updated: {self.task}"

    def call_modify(self, action_input: str) -> str:
        if not os.path.exists(action_input):
            self.history += "observation: file does not exist\n"
            return "File does not exist."

        content = read_python_module_structure(self.directory)[1]
        f_content = (
            content[action_input]
            if content[action_input]
            else "< document is empty >"
        )

        resp = self.run_gpt(
            MODIFY_PROMPT,
            stop_tokens=["action:", "thought:", "observation:"],
            max_tokens=2048,
            task=self.task,
            history=self.history,
            file_path=action_input,
            file_contents=f_content,
        )
        new_contents, description = parse_file_content(resp)
        if new_contents is None:
            self.history += "observation: failed to modify file\n"
            return "Failed to modify file."

        with open(action_input, "w") as f:
            f.write(new_contents)

        self.history += f"observation: file successfully modified\n"
        self.history += f"observation: {description}\n"
        return f"File modified: {action_input}"

    def call_read(self, action_input: str) -> str:
        if not os.path.exists(action_input):
            self.history += "observation: file does not exist\n"
            return "File does not exist."

        content = read_python_module_structure(self.directory)[1]
        f_content = (
            content[action_input]
            if content[action_input]
            else "< document is empty >"
        )

        resp = self.run_gpt(
            READ_PROMPT,
            stop_tokens=[],
            max_tokens=256,
            task=self.task,
            history=self.history,
            file_path=action_input,
            file_contents=f_content,
        ).strip("\n")
        self.history += f"observation: {resp}\n"
        return f"File read: {action_input}"

    def call_add(self, action_input: str) -> str:
        d = os.path.dirname(action_input)
        if not d.startswith(self.directory):
            self.history += (
                f"observation: files must be under directory {self.directory}\n"
            )
            return f"Invalid directory: {d}"
        elif not action_input.endswith(".py"):
            self.history += "observation: can only write .py files\n"
            return "Only .py files are allowed."
        else:
            if d and not os.path.exists(d):
                os.makedirs(d)
            if not os.path.exists(action_input):
                resp = self.run_gpt(
                    ADD_PROMPT,
                    stop_tokens=["action:", "thought:", "observation:"],
                    max_tokens=2048,
                    task=self.task,
                    history=self.history,
                    file_path=action_input,
                )
                new_contents, description = parse_file_content(resp)
                if new_contents is None:
                    self.history += "observation: failed to write file\n"
                    return "Failed to write file."

                with open(action_input, "w") as f:
                    f.write(new_contents)

                self.history += "observation: file successfully written\n"
                self.history += f"observation: {description}\n"
                return f"File added: {action_input}"
            else:
                self.history += "observation: file already exists\n"
                return "File already exists."

    def call_test(self, action_input: str) -> str:
        result = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", self.directory],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.history += f"observation: there are no tests! Test should be written in a test folder under {self.directory}\n"
            return "No tests found."
        result = subprocess.run(
            ["python", "-m", "pytest", self.directory],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            self.history += "observation: tests pass\n"
            return "All tests passed."

        resp = self.run_gpt(
            UNDERSTAND_TEST_RESULTS_PROMPT,
            stop_tokens=[],
            max_tokens=256,
            task=self.task,
            history=self.history,
            stdout=result.stdout[:5000],
            stderr=result.stderr[:5000],
        )
        self.history += f"observation: tests failed: {resp}\n"
        return f"Tests failed: {resp}"


# --- Global Pypelyne Instance ---
pypelyne = Pypelyne()


# --- Helper Functions ---
def create_agent(name: str, agent_type: str, complexity: int) -> Agent:
    agent = Agent(name, agent_type, complexity)
    pypelyne.add_agent(agent)
    return agent


def create_tool(name: str, tool_type: str) -> Tool:
    tool = Tool(name, tool_type)
    pypelyne.add_tool(tool)
    return tool


# --- Streamlit App Code ---
def main():
    st.title("🧠 Pypelyne: Your AI-Powered Coding Assistant")

    # --- Sidebar ---
    st.sidebar.title("⚙️ Settings")
    if "directory" not in st.session_state:
        st.session_state.directory = "."
    pypelyne.directory = st.sidebar.text_input(
        "Project Directory:",
        value=st.session_state.directory,
        help="Path to your coding project",
    )
    st.session_state.directory = pypelyne.directory  # Update session state
    if "purpose" not in st.session_state:
        st.session_state.purpose = ""
    pypelyne.purpose = st.sidebar.text_area(
        "Project Purpose:",
        value=st.session_state.purpose,
        help="Describe the purpose of your coding project.",
    )
    st.session_state.purpose = pypelyne.purpose  # Update session state

    # --- Agent and Tool Management ---
    st.sidebar.header("🤖 Agents")
    if "agents" not in st.session_state:
        st.session_state.agents = []
    show_agent_creation = st.sidebar.expander(
        "Create New Agent", expanded=False
    )
    with show_agent_creation:
        agent_name = st.text_input("Agent Name:")
        agent_type = st.selectbox("Agent Type:", AGENT_TYPES)
        agent_complexity = st.slider("Complexity (1-5):", 1, 5, 3)
        if st.button("Add Agent"):
            create_agent(agent_name, agent_type, agent_complexity)
            st.session_state.agents = pypelyne.agents  # Update session state

    st.sidebar.header("🛠️ Tools")
    if "tools" not in st.session_state:
        st.session_state.tools = []
    show_tool_creation = st.sidebar.expander("Create New Tool", expanded=False)
    with show_tool_creation:
        tool_name = st.text_input("Tool Name:")
        tool_type = st.selectbox("Tool Type:", TOOL_TYPES)
        if st.button("Add Tool"):
            create_tool(tool_name, tool_type)
            st.session_state.tools = pypelyne.tools  # Update session state

    # --- Display Agents and Tools ---
    st.sidebar.subheader("Active Agents:")
    for agent in st.session_state.agents:
        st.sidebar.write(f"- {agent}")

    st.sidebar.subheader("Available Tools:")
    for tool in st.session_state.tools:
        st.sidebar.write(f"- {tool}")

    # --- Main Content Area ---
    st.header("💻 Code Interaction")

    if "task" not in st.session_state:
        st.session_state.task = ""
    task_input = st.text_area(
        "🎯 Task:",
        value=st.session_state.task,
        help="Describe the coding task you want to perform.",
    )
    if task_input:
        pypelyne.task = task_input
        st.session_state.task = pypelyne.task  # Update session state

    user_input = st.text_input(
        "💬 Your Input:", help="Provide instructions or ask questions."
    )

    if st.button("Execute"):
        if user_input:
            with st.spinner("Pypelyne is working..."):
                response = pypelyne.run_action("MAIN", user_input)
                st.write("Pypelyne Says: ", response)


# --- Run the Streamlit app ---
if __name__ == "__main__":
    main()