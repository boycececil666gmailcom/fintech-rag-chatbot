#!/bin/bash

# Load configurations from .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | tr -d '\r' | xargs)
fi

echo "========================================================"
echo "[1/2] Activating Python Virtual Environment"
echo "========================================================"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv || python3 -m venv venv || exit 1
fi

# Activate
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "Installing/checking dependencies..."
pip install -r requirements.txt

echo "========================================================"
echo "[2/2] Running End-to-End Tests"
echo "========================================================"

python -m pytest tests/test_api_e2e.py -v
