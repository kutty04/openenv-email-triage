FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY env/ ./env/
COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Create __init__.py for env package
RUN touch env/__init__.py

# Expose the port HuggingFace Spaces uses
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
