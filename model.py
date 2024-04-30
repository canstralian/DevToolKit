import torch.nn as nn

class CodeGenerator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids):
        return self.model(input_ids)[0]