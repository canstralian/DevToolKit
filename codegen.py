import transformers
from transformers import pipeline

class CodeGenerator:
    def __init__(self, model_name="bigscience/T0_3B"):
        """
        Initializes the CodeGenerator with a specified model.

        Args:
            model_name (str): The name of the model to be used for code generation.
        """
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def generate_code(self, idea):
        """
        Generates code based on a given idea using the specified model.

        Args:
            idea (str): The idea for the code to be generated.

        Returns:
            str: The generated code.
        """
        input_text = self._format_input(idea)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output_sequences = self._generate_output(input_ids)
        generated_code = self._extract_code(output_sequences)

        return generated_code

    def _format_input(self, idea):
        """
        Formats the input text for the model.

        Args:
            idea (str): The idea for the code to be generated.

        Returns:
            str: Formatted input text.
        """
        return f"# Idea: {idea}\n# Code:\n"

    def _generate_output(self, input_ids):
        """
        Generates output sequences from the model.

        Args:
            input_ids (tensor): The input IDs for the model.

        Returns:
            tensor: The generated output sequences.
        """
        return self.model.generate(
            input_ids=input_ids,
            max_length=1024,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.7,
            top_k=50,
        )

    def _extract_code(self, output_sequences):
        """
        Extracts the generated code from the output sequences.

        Args:
            output_sequences (tensor): The generated output sequences.

        Returns:
            str: The extracted code.
        """
        generated_code = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return generated_code.split("\n# Code:")[1].strip()

# Example usage
if __name__ == "__main__":
    idea = "Write a Python function to calculate the factorial of a number"
    code_generator = CodeGenerator()
    generated_code = code_generator.generate_code(idea)
    print(generated_code)