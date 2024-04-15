FROM python:3.8

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set environment variables (optional)
# ENV HF_TOKEN="write_token"

# Download and cache the code generation model
RUN transformers-cli login
RUN transformers-cli download Salesforce/codegen-350M-mono --cache_dir /app/.cache

# Expose port for Gradio interface (optional)
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]