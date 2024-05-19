from flask import Flask, request, jsonify
from huggingface_hub import HfApi
from transformers import pipeline

app = Flask(__name__)
api = HfApi()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

@app.route('/search_datasets', methods=['GET'])
def search_datasets():
    try:
        query = request.args.get('query')
        if not query:
            logger.error("No query provided for dataset search.")
            return jsonify({"error": "No query parameter provided"}), 400
        
        logger.info(f"Searching datasets with query: {query}")
        datasets = api.list_datasets(search=query, full=True)
        return jsonify(datasets)
    except Exception as e:
        logger.error(f"Failed to search datasets: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/run_inference', methods=['POST'])
def run_inference():
    try:
        model_id = request.json.get('model_id')
        inputs = request.json.get('inputs')
        
        if not model_id or not inputs:
            logger.error("Model ID or inputs missing in the request.")
            return jsonify({"error": "Model ID or inputs missing in the request"}), 400
        
        logger.info(f"Running inference using model: {model_id}")
        model_pipeline = pipeline(task="text-generation", model=model_id)
        results = model_pipeline(inputs)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Failed to run inference: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.launch()