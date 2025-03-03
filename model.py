
import torch.nn as nn
from transformers import AutoModelForCausalLM

class CodeGenerator(nn.Module):
    """
    A PyTorch module that generates code using a pre-trained language model.

    This class inherits from `nn.Module` and encapsulates a pre-trained language model
    from the Hugging Face Transformers library. The model is used to generate code
    based on the input sequence.

    Attributes:
    - model (transformers.AutoModelForCausalLM): The pre-trained language model
                                                used for code generation.
    """
    def __init__(self, model_name):
        """
        Initializes a new instance of the `CodeGenerator` class.

        Parameters:
        - model_name (str): The name of the pre-trained language model to use.
                           This should be a valid model name from the Hugging Face
                           Transformers library.
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids):
        """
        Generates code based on the input sequence.

        Parameters:
        - input_ids (torch.Tensor): A tensor of token IDs representing the input
                                   sequence for the language model.

        Returns:
        torch.Tensor: The output tensor containing the generated code.
        """
        return self.model(input_ids)[0]
