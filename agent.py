
import os
import random
import logging
import gradio as gr
import asyncio
from typing import List, Tuple, Generator, Any
from inference_client import InferenceClient  # Adjust the import as needed

# Set up logging to capture errors and warnings.
logging.basicConfig(
    level=logging.INFO,
    filename='chatbot.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Encapsulated configuration to avoid global variable pitfalls.
class ChatbotConfig:
    def __init__(
        self,
        max_history: int = 100,
        verbose: bool = True,
        max_iterations: int = 1000,
        max_new_tokens: int = 256,
        default_seed: int = None
    ):
        self.max_history = max_history
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.max_new_tokens = max_new_tokens
        self.default_seed = default_seed or random.randint(1, 2**32 - 1)

# Global configuration instance.
config = ChatbotConfig()

# Externalize prompts into a dictionary, optionally overridden by environment variables.
PROMPTS = {
    "ACTION_PROMPT": os.environ.get("ACTION_PROMPT", "action prompt"),
    "ADD_PROMPT": os.environ.get("ADD_PROMPT", "add prompt"),
    "COMPRESS_HISTORY_PROMPT": os.environ.get("COMPRESS_HISTORY_PROMPT", "compress history prompt"),
    "LOG_PROMPT": os.environ.get("LOG_PROMPT", "log prompt"),
    "LOG_RESPONSE": os.environ.get("LOG_RESPONSE", "log response"),
    "MODIFY_PROMPT": os.environ.get("MODIFY_PROMPT", "modify prompt"),
    "PREFIX": os.environ.get("PREFIX", "prefix"),
    "SEARCH_QUERY": os.environ.get("SEARCH_QUERY", "search query"),
    "READ_PROMPT": os.environ.get("READ_PROMPT", "read prompt"),
    "TASK_PROMPT": os.environ.get("TASK_PROMPT", "task prompt"),
    "UNDERSTAND_TEST_RESULTS_PROMPT": os.environ.get("UNDERSTAND_TEST_RESULTS_PROMPT", "understand test results prompt")
}

# Instantiate the AI client.
client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def format_prompt_var(message: str, history: List[str]) -> str:
    """
    Format the provided message and conversation history into the required prompt format.

    Args:
        message (str): The current instruction/message.
        history (List[str]): List of previous conversation entries.
    
    Returns:
        str: A formatted prompt string.
    
    Raises:
        TypeError: If message is not a string or any history entry is not a string.
    """
    if not isinstance(message, str):
        raise TypeError("The instruction message must be a string.")
    if not all(isinstance(item, str) for item in history):
        raise TypeError("All items in history must be strings.")
    
    history_text = "\n".join(history) if history else "No previous conversation."
    prompt = f"\n### Instruction:\n{message}\n### History:\n{history_text}"
    return prompt

def run_agent(instruction: str, history: List[str]) -> Tuple[str, List[str]]:
    """
    Run the AI agent with the given instruction and conversation history.

    Args:
        instruction (str): The user instruction.
        history (List[str]): The conversation history.
    
    Returns:
        Tuple[str, List[str]]: A tuple containing the full AI response and a list of extracted actions.
    
    Raises:
        TypeError: If inputs are of invalid type.
    """
    if not isinstance(instruction, str):
        raise TypeError("Instruction must be a string.")
    if not isinstance(history, list) or not all(isinstance(item, str) for item in history):
        raise TypeError("History must be a list of strings.")

    prompt = format_prompt_var(instruction, history)
    response = ""
    iterations = 0

    try:
        for chunk in generate(prompt, history[-config.max_history:], temperature=0.7):
            response += chunk
            iterations += 1
            if "\n\n### Instruction:" in chunk or iterations >= config.max_iterations:
                break
    except Exception as e:
        logging.error("Error in run_agent: %s", e)
        response += f"\n[Error in run_agent: {e}]"

    # Extract actions from the response.
    response_actions = []
    for line in response.strip().split("\n"):
        if line.startswith("action:"):
            response_actions.append(line.replace("action: ", ""))
    
    return response, response_actions

def generate(prompt: str, history: List[str], temperature: float) -> Generator[str, None, None]:
    """
    Generate text from the AI model using the formatted prompt.
    
    Args:
        prompt (str): The input prompt.
        history (List[str]): Recent conversation history.
        temperature (float): Sampling temperature.
    
    Yields:
        str: Incremental output from the text-generation stream.
    """
    seed = random.randint(1, 2**32 - 1) if config.default_seed is None else config.default_seed
    generate_kwargs = {
        "temperature": temperature,
        "max_new_tokens": config.max_new_tokens,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "seed": seed,
    }
    formatted_prompt = format_prompt_var(prompt, history)
    
    try:
        stream = client.text_generation(
            formatted_prompt,
            **generate_kwargs,
            stream=True,
            details=True,
            return_full_text=False
        )
    except Exception as e:
        logging.error("Error during text_generation call: %s", e)
        yield f"[Error during text_generation call: {e}]"
        return

    output = ""
    iterations = 0
    for response in stream:
        iterations += 1
        try:
            output += response.token.text
        except AttributeError as ae:
            logging.error("Malformed response token: %s", ae)
            yield f"[Malformed response token: {ae}]"
            break
        yield output
        if iterations >= config.max_iterations:
            yield "\n[Response truncated due to length limitations]"
            break

async def async_run_agent(instruction: str, history: List[str]) -> Tuple[str, List[str]]:
    """
    Asynchronous wrapper to run the agent in a separate thread.
    
    Args:
        instruction (str): The instruction for the AI.
        history (List[str]): The conversation history.
    
    Returns:
        Tuple[str, List[str]]: The response and extracted actions.
    """
    return await asyncio.to_thread(run_agent, instruction, history)

def clear_conversation() -> List[str]:
    """
    Clear the conversation history.
    
    Returns:
        List[str]: An empty conversation history.
    """
    return []

def update_chatbot_styles(history: List[Any]) -> Any:
    """
    Update the chatbot display styles based on the number of messages.

    Args:
        history (List[Any]): The current conversation history.
    
    Returns:
        Update object for Gradio Chatbot.
    """
    num_messages = sum(1 for item in history if isinstance(item, tuple))
    return gr.Chatbot.update({"num_messages": num_messages})

def update_max_history(value: int) -> int:
    """
    Update the max_history in configuration.

    Args:
        value (int): New maximum history value.
    
    Returns:
        int: The updated max_history.
    """
    config.max_history = int(value)
    return config.max_history

def create_interface() -> gr.Blocks:
    """
    Create and return the Gradio interface for the chatbot application.
    
    Returns:
        gr.Blocks: The Gradio Blocks object representing the UI.
    """
    block = gr.Blocks()
    chatbot = gr.Chatbot()

    with block:
        gr.Markdown("## Expert Web Developer Assistant")
        with gr.Tab("Conversation"):
            txt = gr.Textbox(show_label=False, placeholder="Type something...")
            btn = gr.Button("Send", variant="primary")
            
            # When text is submitted, run the agent asynchronously.
            txt.submit(
                async_run_agent, 
                inputs=[txt, chatbot], 
                outputs=[chatbot, None]
            )
            # Clear conversation history and update chatbot UI.
            txt.clear(fn=clear_conversation, outputs=chatbot).then(
                update_chatbot_styles, chatbot, chatbot
            )
            btn.click(fn=clear_conversation, outputs=chatbot).then(
                update_chatbot_styles, chatbot, chatbot
            )

        with gr.Tab("Settings"):
            max_history_slider = gr.Slider(
                minimum=1, maximum=100, step=1,
                label="Max history", 
                value=config.max_history
            )
            max_history_slider.change(
                update_max_history, max_history_slider, max_history_slider
            )

    return block

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()