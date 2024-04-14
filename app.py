import gradio as gr
from transformers import pipeline

from components.database_page import database_page
from components.documentation_page import documentation_page
from components.home import home
from components.lang_page import lang_page
from components.optimization_page import optimization_page
from components.refactor_page import refactor_page
from components.style_page import style_page
from components.test_page import test_page

# Gradio interface
def setup_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Select Model and Task")
        with gr.Row():
            model_name = gr.Dropdown(label="Model", choices=["gpt2", "bert-base-uncased"])
            task = gr.Dropdown(label="Task", choices=["text-generation", "text-classification"])
        input_data = gr.Textbox(label="Input")
        output = gr.Textbox(label="Output")

        input_data.change(fn=model_inference, inputs=[model_name, task, input_data], outputs=output)

    return demo

# Function to generate text or perform other tasks based on model selection
def model_inference(model_name, task, input_data):
    try:
        model_pipeline = pipeline(task, model=model_name)
        result = model_pipeline(input_data)
        return result
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    interface = setup_interface()
    interface.launch()