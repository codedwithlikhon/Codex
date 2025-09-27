# Use a slim Python base image
FROM python:3.11-slim

# Install curl for downloading ollama
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install ollama
RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama && chmod +x /usr/bin/ollama

# Pull the AI model into the image. This makes it available at runtime without a delay.
# Using gemma:2b as a small, capable model for code analysis.
RUN ollama pull gemma:2b

# Copy the application code into the container
WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ollama API port (optional, but good practice)
EXPOSE 11434

# Define the entrypoint to start ollama and run the review script
# This script will start ollama in the background and then execute the main review logic
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "review_pr.py"]