#!/bin/bash
# Startup script for ERT LLM Backend on Vast.ai

set -e

echo "Starting ERT LLM Backend services..."

# Start Ollama service in background
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
sleep 10

# Pull the configured model (default: GPT-OSS:20b)
MODEL_NAME=${OLLAMA_MODEL:-gpt-oss:20b}
echo "Pulling model: $MODEL_NAME"
ollama pull $MODEL_NAME

# Verify model is available
echo "Verifying model availability..."
ollama list

# Start FastAPI backend
echo "Starting FastAPI backend..."
python3 llm_backend.py

# If FastAPI exits, cleanup
echo "Cleaning up..."
kill $OLLAMA_PID 2>/dev/null || true
exit 0