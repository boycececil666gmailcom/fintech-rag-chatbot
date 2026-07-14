#!/bin/bash

# Load configurations from .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | tr -d '\r' | xargs)
fi

echo -e "\n\033[1;96m========================================================\033[0m"
echo -e "\033[1;92m>>> [1/2] [$(basename "$0")] Activating Python Virtual Environment\033[0m"
echo -e "\033[1;96m========================================================\033[0m\n"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv || python3 -m venv venv || exit 1
    
    # Activate
    if [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    # Activate
    if [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

echo -e "\n\033[1;96m========================================================\033[0m"
echo -e "\033[1;92m>>> [2/2] [$(basename "$0")] Running Unit Tests\033[0m"
echo -e "\033[1;96m========================================================\033[0m\n"

python -m pytest tests/test_api_unit.py -v
