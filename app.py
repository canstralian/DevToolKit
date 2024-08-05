import os
import subprocess
from huggingface_hub import InferenceClient
import gradio as gr
import random
import time
from typing import List, Dict
from flask import Flask, request, jsonify

# Constants
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
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Initialize Hugging Face client
client = InferenceClient(MODEL)

# Import necessary prompts and functions from the existing code
from .prompts import (
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
from .utils import parse_action, parse_file_content, read_python_module_structure

class Agent:
    def __init__(self, name: str, agent_type: str, complexity: int):
        self.name = name
        self.type = agent_type
        self.complexity = complexity
        self.tools = []

    def add_tool(self, tool):
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
        self.history = ""
        self.task = None
        self.purpose = None
        self.directory = None

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def add_tool(self, tool: Tool):
        self.tools.append(tool)

    def generate_chat_app(self):
        time.sleep(2)  # Simulate processing time
        return f"Chat app generated with {len(self.agents)} agents and {len(self.tools)} tools."

    def run_gpt(self, prompt_template, stop_tokens, max_tokens, **prompt_kwargs):
        content = PREFIX.format(
            module_summary=read_python_module_structure(self.directory)[0],
            purpose=self.purpose,
        ) + prompt_template.format(**prompt_kwargs)

        if VERBOSE:
            print(LOG_PROMPT.format(content))

        stream = client.text_generation(
            prompt=content,
            max_new_tokens=max_tokens,
            stop_sequences=stop_tokens if stop_tokens else None,
            do_sample=True,
            temperature=0.7,
        )

        resp = "".join(token for token in stream)

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

    def run_action(self, action_name, action_input):
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

    def call_main(self, action_input):
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

    def call_set_task(self, action_input):
        self.task = self.run_gpt(
            TASK_PROMPT,
            stop_tokens=[],
            max_tokens=64,
            task=self.task,
            history=self.history,
        ).strip("\n")
        self.history += f"observation: task has been updated to: {self.task}\n"
        return f"Task updated: {self.task}"

    def call_modify(self, action_input):
        if not os.path.exists(action_input):
            self.history += "observation: file does not exist\n"
            return "File does not exist."

        content = read_python_module_structure(self.directory)[1]
        f_content = (
            content[action_input] if content[action_input] else "< document is empty >"
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

    def call_read(self, action_input):
        if not os.path.exists(action_input):
            self.history += "observation: file does not exist\n"
            return "File does not exist."

        content = read_python_module_structure(self.directory)[1]
        f_content = (
            content[action_input] if content[action_input] else "< document is empty >"
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

    def call_add(self, action_input):
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

    def call_test(self, action_input):
        result = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", self.directory],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.history += f"observation: there are no tests! Test should be written in a test folder under {self.directory}\n"
            return "No tests found."
        result = subprocess.run(
            ["python", "-m", "pytest", self.directory], capture_output=True, text=True
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

pypelyne = Pypelyne()

def create_agent(name: str, agent_type: str, complexity: int) -> Agent:
    agent = Agent(name, agent_type, complexity)
    pypelyne.add_agent(agent)
    return agent

def create_tool(name: str, tool_type: str) -> Tool:
    tool = Tool(name, tool_type)
    pypelyne.add_tool(tool)
    return tool

def main():
    # Create a Flask app
    app = Flask(__name__)

    # Define a route for the chat interface
    @app.route("/chat", methods=["GET", "POST"])
    def chat():
        if request.method == "POST":
            # Get the user's input
            user_input = request.form["input"]

            # Run the input through the Pypelyne
            response = pypelyne.run_action("MAIN", user_input)

            # Return the response
            return jsonify({"response": response})
        else:
            # Return the chat interface
            return """
                <html>
                    <body>
                        <h1>Pypelyne Chat Interface</h1>
                        <form action="/chat" method="post">
                            <input type="text" name="input" placeholder="Enter your input">
                            <input type="submit" value="Submit">
                        </form>
                        <div id="response"></div>
                        <script>
                            // Update the response div with the response from the server
                            function updateResponse(response) {
                                document.getElementById("response").innerHTML = response;
                            }
                        </script>
                    </body>
                </html>
            """

    # Define a route for the agent creation interface
    @app.route("/create_agent", methods=["GET", "POST"])
    def create_agent_interface():
        if request.method == "POST":
            # Get the agent's name, type, and complexity
            name = request.form["name"]
            agent_type = request.form["type"]
            complexity = int(request.form["complexity"])

            # Create the agent
            agent = create_agent(name, agent_type, complexity)

            # Return a success message
            return jsonify({"message": f"Agent {name} created successfully"})
        else:
            # Return the agent creation interface
            return """
                <html>
                    <body>
                        <h1>Create Agent</h1>
                        <form action="/create_agent" method="post">
                            <label for="name">Name:</label>
                            <input type="text" id="name" name="name"><br><br>
                            <label for="type">Type:</label>
                            <select id="type" name="type">
                                <option value="Task Executor">Task Executor</option>
                                <option value="Information Retriever">Information Retriever</option>
                                <option value="Decision Maker">Decision Maker</option>
                                <option value="Data Analyzer">Data Analyzer</option>
                            </select><br><br>
                            <label for="complexity">Complexity:</label>
                            <input type="number" id="complexity" name="complexity"><br><br>
                            <input type="submit" value="Create Agent">
                        </form>
                    </body>
                </html>
            """

    # Define a route for the tool creation interface
    @app.route("/create_tool", methods=["GET", "POST"])
    def create_tool_interface():
        if request.method == "POST":
            # Get the tool's name and type
            name = request.form["name"]
            tool_type = request.form["type"]

            # Create the tool
            tool = create_tool(name, tool_type)

            # Return a success message
            return jsonify({"message": f"Tool {name} created successfully"})
        else:
            # Return the tool creation interface
            return """
                <html>
                    <body>
                        <h1>Create Tool</h1>
                        <form action="/create_tool" method="post">
                            <label for="name">Name:</label>
                            <input type="text" id="name" name="name"><br><br>
                            <label for="type">Type:</label>
                            <select id="type" name="type">
                                <option value="Web Scraper">Web Scraper</option>
                                <option value="Database Connector">Database Connector</option>
                                <option value="API Caller">API Caller</option>
                                <option value="File Handler">File Handler</option>
                                <option value="Text Processor">Text Processor</option>
                            </select><br><br>
                            <input type="submit" value="Create Tool">
                        </form>
                    </body>
                </html>
            """

    # Run the app
    if __name__ == "__main__":
        app.run(debug=True)

if __name__ == "__main__":
    main()