import transformers

class CodeTokenizer(transformers.AutoTokenizer):
    def __init__(self, model_name):
        super().__init__.from_pretrained(model_name)