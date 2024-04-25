from flask import Flask, request, jsonify
from huggingface_hub import HfApi
import streamlit

app = Flask(__name__)
api = HfApi()

@app.route('/search_datasets', methods=['GET'])
def search_datasets():
    query = request.args.get('query')
    datasets = api.list_datasets(search=query, full=True)
    return jsonify(datasets)

@app.route('/run_inference', methods=['POST'])
def run_inference():
    model_id = request.json['model_id']
    inputs = request.json['inputs']
    # Assuming the model is compatible with the pipeline API
    from transformers import pipeline
    model_pipeline = pipeline(task="text-generation", model=model_id)
    results = model_pipeline(inputs)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False)