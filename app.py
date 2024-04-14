from huggingface_hub import HfApi
from transformers import pipeline

# Function to merge models at equal weights
def merge_models(models):
    model_weights = [1.0 / len(models) for _ in range(len(models)]
    merged_model = pipeline("text-generation", model=models, model_weights=model_weights)
    return merged_model

# Retrieve code-generative models with config.json
def get_code_generative_models():
    api = HfApi()
    models_list = api.list_models()

    code_generative_models = []

    for model in models_list:
        model_id = model.modelId
        model_info = api.model_info(model_id)

        if "config.json" in model_info.keys():
            code_generative_models.append(model_id)

    return code_generative_models

# Main function to merge models and deploy the merged model
def main():
    code_generative_models = get_code_generative_models()
    
    if len(code_generative_models) < 2:
        print("At least two code-generative models with config.json files are required for merging.")
        return

    models = [model for model in code_generative_models[:2]]  # Select the first two models for merging
    merged_model = merge_models(models)

    # Embed the merged model into a chat app for testing
    chat_app = pipeline("text-generation", model=merged_model)

    # Provide options for the user to download the code/config or deploy the merged model
    print("Chat App Ready for Testing!")
    print("Options:")
    print("1. Download Code/Config")
    print("2. Deploy as a Unique Space (Requires Write-Permission API Key)")

    user_choice = input("Enter your choice (1 or 2): ")

    if user_choice == "1":
        # Download code/config
        merged_model.save_pretrained("merged_model")

    elif user_choice == "2":
        # Deploy as a Unique Space with write-permission API Key
        api_key = input("Enter your write-permission API Key: ")
        # Code to deploy the merged model using the provided API key

if __name__ == "__main__":
    main()