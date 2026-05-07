#!/bin/bash

# =============================================================================
# Pragya Development Environment Starter
# =============================================================================

echo "🔄 Starting Ollama server..."
# Start Ollama service in the background
ollama serve &
OLLAMA_PID=$!

# Wait a few seconds to let Ollama initialize properly
echo "⏳ Waiting for Ollama to initialize..."
sleep 3

echo "🚀 Starting Streamlit application..."
# Run the Streamlit app from the virtual environment
./venv/Scripts/python.exe -m streamlit run app.py

# When the Streamlit app is closed (e.g., via Ctrl+C), kill the Ollama server
echo "🛑 Stopping Ollama server..."
kill $OLLAMA_PID
echo "✅ Done."
