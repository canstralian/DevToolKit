import gradio as gr
from transformers import pipeline

# Load the text generation pipeline
pipe = pipeline(
    "text-generation", 
    model="MaziyarPanahi/BASH-Coder-Mistral-7B-Mistral-7B-Instruct-v0.2-slerp-GGUF"
)

def generate_bash_code(prompt):
    """Generates BASH code using the Mistral-7B pipeline."""
    sequences = pipe(
        prompt, 
        max_length=200, 
        num_return_sequences=1,
        do_sample=True,  # Enable sampling for more creative output
        top_k=50,       # Explore a wider range of vocabulary
        top_p=0.95,      # Control the probability distribution of tokens
        temperature=0.8   # Adjust temperature for creativity 
    )
    return sequences[0]['generated_text']

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_bash_code,
    inputs=gr.Textbox(lines=5, label="Describe what you want your BASH script to do"),
    outputs=gr.Code(language="bash", label="Generated BASH Code"),
    title="BASH Coder",
    description="Generate BASH scripts using a Mistral-7B model.",
)

# Launch the interface
iface.launch()