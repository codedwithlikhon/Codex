#!/bin/sh
# Start the ollama server in the background
ollama serve &

# Wait a few seconds for the server to start
sleep 5

# Execute the command passed to the container (e.g., python review_pr.py)
exec "$@"