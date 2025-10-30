FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install transformers and CPU-only PyTorch wheels
RUN pip install --no-cache-dir transformers torch --extra-index-url https://download.pytorch.org/whl/cpu

# Copy and execute model download script
COPY download_with_hf.py .
RUN python download_with_hf.py && rm download_with_hf.py

# Copy application code
COPY api/ .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

