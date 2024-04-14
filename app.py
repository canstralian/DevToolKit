import os
import gradio as gr
from transformers import AutoModel, AutoTokenizer

   os.environ["GRADIO_SERVER_PORT"] = "8507"

def get_code_generative_models():
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    models = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            model_info = AutoModel.from_pretrained(model_path)
            if "config.json" in [f.name for f in model_info.files]:
                models.append((model_name, model_path))
    return models

def model_inference(model_name, model_path, input_data):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    inputs = tokenizer(input_data, return_tensors="pt")
    outputs = model(**inputs)
    result = outputs.last_hidden_state[:, 0, :]
    return result.tolist()

def main():
    models = get_code_generative_models()
    with gr.Blocks() as demo:
        gr.Markdown("### Select Model and Input")
        with gr.Row():
            model_name = gr.Dropdown(label="Model", choices=[m[0] for m in models])
            input_data = gr.Textbox(label="Input")

        model_path = gr.State(None)

        def update_model_path(model_name):
            model_path.set(next(filter(lambda m: m[0] == model_name, models))[1])

        input_data.change(update_model_path, inputs=model_name, outputs=model_path)

        output = gr.Textbox(label="Output")

        def infer(model_name, input_data):
            return model_inference(model_name, model_path, input_data)

        output.change(fn=infer, inputs=[model_name, input_data], outputs=output)

    interface = demo.launch(server_port=get_free_port())

if __name__ == "__main__":
    main()