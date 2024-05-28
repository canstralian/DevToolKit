from transformers import pipeline

def generate_code_from_model(input_text):
    """Generates code using the bigscience/T0_3B model."""
    model_name = 'bigscience/T0_3B'  # Choose your model
    generator = pipeline('text-generation', model=model_name)
    code = generator(input_text, max_length=50, num_return_sequences=1, do_sample=True)[0]['generated_text']
    return code