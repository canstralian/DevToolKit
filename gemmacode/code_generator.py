import transformers

def generate(idea):
    # Load the code generation model
    model_name = "Salesforce/codegen-350M-mono"
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
    )
    generated_code = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return generated_code