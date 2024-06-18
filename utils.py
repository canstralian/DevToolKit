import ast
import glob
import os


def parse_file_content(string: str) -> tuple[str | None, str | None]:
    """
    Parses the content of a file and returns the action and description.
    :param string: The content of a file.
    :return: A tuple containing the action and description.
    """
    first_break = string.find("---")
    last_break = string.rfind("---")
    if first_break == -1 and last_break == -1 or first_break == last_break:
        return None, None
    
    # Find the newline after the last separator
    nl_after = string.find("\n", last_break) + 1
    description = string[nl_after:].strip() if nl_after > last_break else ""

    return string[first_break + 4 : last_break], description


def parse_action(string: str) -> tuple[str, str | None]:
    """
    Parses the action from a string.
    :param string: The string to parse the action from.
    :return: A tuple containing the action and action input.
    """
    assert string.startswith("action:")
    idx = string.find("action_input=")
    action = string[8: idx - 1] if idx > 8 else string[8:]
    action_input = string[idx + 13 :].strip("'").strip('"') if idx > 8 else None
    return action, action_input


def extract_imports(file_contents: str) -> tuple[list[str], list[ast.FunctionDef], list[ast.ClassDef]]:
    """
    Extracts imports, functions, and classes from a file's contents.
    :param file_contents: The contents of a file.
    :return: A tuple containing the imports, functions, and classes.
    """
    module_ast = ast.parse(file_contents)
    imports = []
    functions = [n for n in module_ast.body if isinstance(n, ast.FunctionDef)]
    classes = [n for n in module_ast.body if isinstance(n, ast.ClassDef)]

    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            for alias in node.names:
                name = alias.name
                if module_name:
                    imports.append(f"{module_name}.{name}")
                else:
                    imports.append(name)

    return imports, functions, classes


def read_python_module_structure(path: str) -> tuple[str, dict[str, str], dict[str, list[str]]]:
    """
    Reads the structure of a Python module and returns a prompt, content, and internal imports map.
    :param path: The path to the Python module.
    :return: A tuple containing the structure prompt, content, and internal imports map.
    """
    file_types = ["*.py"]
    code = []
    for file_type in file_types:
        code += glob.glob(os.path.join(path, "**", file_type), recursive=True)

    structure_prompt = "Files:\n"
    structure_prompt += "(listing all files and their functions and classes)\n\n"

    def get_file_name(i: str) -> str:
        return "./{}.py".format(i.replace(".", "/"))

    content = {}
    internal_imports_map = {}
    for fn in code:
        if os.path.basename(fn) == "gpt.py":
            continue
        with open(fn, "r") as f:
            content[fn] = f.read()

        imports, functions, classes = extract_imports(content[fn])
        internal_imports = [
            ".".join(i.split(".")[:-1])
            for i in imports
            if i.startswith("app.")
        ]
        internal_imports_map[fn] = [
            get_file_name(i) for i in set(internal_imports)
        ]

        structure_prompt += f"{fn}\n"
        for function in functions:
            structure_prompt += f"  {function.name}({function.args})\n"

    return structure_prompt, content, internal_imports_map