
import transformers
from transformers import pipeline

def generate(idea):
    """
    Generates code based on a given idea using the bigscience/T0_3B model.

    Args:
        idea (str): The idea for the code to be generated.

    Returns:
        str: The generated code.
    """
    # Load the code generation model
    model_name = "bigscience/T0_3B"  # Use a model that works for code generation
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # Generate the code
    input_text = f"""
    # Idea: {idea}
    # Code:
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=1024,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7,  # Adjust temperature for creativity
        top_k=50,  # Adjust top_k for diversity
    )
    generated_code = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Remove the prompt and formatting
    generated_code = generated_code.split("\n# Code:")[1].strip()

    return generated_code

# Example usage
idea = "Write a Python function to calculate the factorial of a number"
code = generate(idea)
print(code)
