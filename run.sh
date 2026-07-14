#!/bin/bash

# Load configurations from .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi
OLLAMA_MODEL=${OLLAMA_MODEL:-"qwen2.5:7b"}

echo -e "\n\033[1;96m========================================================\033[0m"
echo -e "\033[1;92m>>> [1/3] [$(basename "$0")] Setting Up Python Virtual Environment\033[0m"
echo -e "\033[1;96m========================================================\033[0m\n"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo -e "\n\033[1;96m========================================================\033[0m"
echo -e "\033[1;92m>>> [2/3] [$(basename "$0")] Checking Ollama Server and Model\033[0m"
echo -e "\033[1;96m========================================================\033[0m\n"

# Verify Ollama is running
echo "Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama is not running on http://localhost:11434. Please start Ollama first."
    exit 1
fi
echo "Ollama server is active."

# Ensure OLLAMA_MODEL is downloaded
if ! curl -s http://localhost:11434/api/tags | grep -q "\"$OLLAMA_MODEL\""; then
    echo "Model '$OLLAMA_MODEL' not found. Pulling from Ollama..."
    ollama pull "$OLLAMA_MODEL" || exit 1
fi
echo "Model '$OLLAMA_MODEL' is ready."

echo -e "\n\033[1;96m========================================================\033[0m"
echo -e "\033[1;92m>>> [3/3] [$(basename "$0")] Installing Dependencies and Starting Server\033[0m"
echo -e "\033[1;96m========================================================\033[0m\n"

pip install -r requirements.txt

echo "Starting backend server..."
python -m src.main
