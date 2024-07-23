import os
import sys
import time

import huggingface_hub

import transformers

from transformers import pipeline

import gradio as gr

import tempfile

from huggingface_hub import HfFolder

def main():
    # Get the user's idea
    idea = input("What is your idea for an application? ")

    # Generate the code for the application
    code = gemmacode.generate(idea)

    # Test the code
    try:
        transformers.pipeline("text-generation")(code)
    except Exception as e:
        print("The code failed to run:", e)
        return

    # Ensure the functionality of the application
    try:
        gr.Interface(fn=transformers.pipeline("text-generation"), inputs=gr.Textbox(), outputs=gr.Textbox()).launch()
    except Exception as e:
        print("The application failed to run:", e)
        return

    # Provide an embedded webapp demo of the user's idea implementation
    try:
        hf_folder = HfFolder(path=tempfile.mkdtemp())
        hf_folder.save(code)
        hf_folder.push_to_hub(repo_id="acecalisto3/gemmacode-demo", commit_message="Initial commit")
        print(f"The demo is available at: https://huggingface.co/acecalisto3/gemmacode-demo")
    except Exception as e:
        print("The demo failed to launch:", e)
        return

    # Offer the option to rebuild or deploy
    while True:
        choice = input("Do you want to rebuild or deploy the application? (r/d/q) ")
        if choice == "r":
            # Rebuild the code
            code = gemmacode.generate(idea)

            # Test the code
            try:
                transformers.pipeline("text-generation")(code)
            except Exception as e:
                print("The code failed to run:", e)
                return

            # Ensure the functionality of the application
            try:
                gr.Interface(fn=transformers.pipeline("text-generation"), inputs=gr.Textbox(), outputs=gr.Textbox()).launch()
            except Exception as e:
                print("The application failed to run:", e)
                return

            # Provide an embedded webapp demo of the user's idea implementation
            try:
                hf_folder = HfFolder(path=tempfile.mkdtemp())
                hf_folder.save(code)
                hf_folder.push_to_hub(repo_id="acecalisto3/gemmacode-demo", commit_message="Initial commit")
                print(f"The demo is available at: https://huggingface.co/acecalisto3/gemmacode-demo")
            except Exception as e:
                print("The demo failed to launch:", e)
                return
        elif choice == "d":
            # Deploy the application
            try:
                api_token = os.environ["HF_TOKEN"]
                hub = huggingface_hub.HfApi(api_token=api_token)
                hub.create_repo(name="my-app", organization="my-org")
                hf_folder = HfFolder(path=tempfile.mkdtemp())
                hf_folder.save(code)
                hf_folder.push_to_hub(repo_id="my-org/my-app", commit_message="Initial commit")
                print("The application has been deployed to: https://huggingface.co/my-org/my-app")
            except Exception as e:
                print("The application failed to deploy:", e)
                return
        elif choice == "q":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()