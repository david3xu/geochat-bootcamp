# Module 2: AI Engine with Snowflake Cortex
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/ai_api.py
ENV FLASK_ENV=development

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model for geological text processing
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/{embeddings,models,logs}

# Expose port for AI API
EXPOSE 5001

# Health check for AI service monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/api/ai/health || exit 1

# Run the AI service
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5001"]
