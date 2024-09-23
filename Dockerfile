# Use a slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose necessary ports
EXPOSE 8000 8501

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Command to start the services
CMD ["sh", "-c", "ollama serve & sleep 10 && ollama pull mxbai-embed-large && ollama pull qwen2.5:1.5b && ollama run llava:latest && uvicorn api:app --host 0.0.0.0 --port 8000 --reload & streamlit run apps.py --server.port 8501"]
