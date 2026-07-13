#!/bin/bash

echo "========================================================"
echo "[1/4] Checking Port 8000 Availability and Ollama Server"
echo "========================================================"
# Port check
if command -v lsof >/dev/null 2>&1; then
    if lsof -i :8000 >/dev/null 2>&1; then
        echo "Error: Port 8000 is already in use. Please free port 8000 before running."
        exit 1
    fi
elif command -v netstat >/dev/null 2>&1; then
    if netstat -ano | grep -q "LISTENING.*:8000"; then
        echo "Error: Port 8000 is already in use. Please free port 8000 before running."
        exit 1
    fi
fi

# Load OLLAMA_MODEL from .env or default to qwen2.5:1.5b
OLLAMA_MODEL="qwen2.5:1.5b"
if [ -f ".env" ]; then
    ENV_MODEL=$(grep -E "^OLLAMA_MODEL=" .env | cut -d'=' -f2 | tr -d '\r' | tr -d '"' | tr -d "'")
    if [ ! -z "$ENV_MODEL" ]; then
        OLLAMA_MODEL="$ENV_MODEL"
    fi
fi

# Ollama check
echo "Checking if Ollama is running..."
TAGS_JSON=$(curl -s http://localhost:11434/api/tags)
if [ $? -ne 0 ] || [ -z "$TAGS_JSON" ]; then
    echo "Error: Ollama is not running on http://localhost:11434. Please start Ollama first."
    exit 1
fi
echo "Ollama server is active."

# Verify model availability
if echo "$TAGS_JSON" | grep -q "\"$OLLAMA_MODEL\""; then
    echo "Model '$OLLAMA_MODEL' is locally available."
else
    echo "Model '$OLLAMA_MODEL' is not downloaded. Pulling '$OLLAMA_MODEL' from Ollama library..."
    ollama pull "$OLLAMA_MODEL"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to pull model '$OLLAMA_MODEL' from Ollama."
        exit 1
    fi
    echo "Model '$OLLAMA_MODEL' downloaded successfully."
fi

# Check if model is loaded in memory
PS_JSON=$(curl -s http://localhost:11434/api/ps)
if echo "$PS_JSON" | grep -q "\"$OLLAMA_MODEL\""; then
    echo "Status: Model '$OLLAMA_MODEL' is currently loaded in memory."
else
    echo "Status: Model '$OLLAMA_MODEL' is ready (will auto-load on first query)."
fi

echo "========================================================"
echo "[2/4] Setting Up Python Virtual Environment"
echo "========================================================"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists."
fi

# Determine virtual environment activation command
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "========================================================"
echo "[3/4] Installing Dependencies"
echo "========================================================"
pip install --upgrade pip
pip install -r requirements.txt

echo "========================================================"
echo "[4/4] Starting FastAPI Server"
echo "========================================================"
echo "Starting backend server on port 8000..."
python -m src.main
