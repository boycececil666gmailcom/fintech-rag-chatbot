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
    VENV_PYTHON="venv/Scripts/python"
    VENV_PIP="venv/Scripts/pip"
else
    source venv/bin/activate
    VENV_PYTHON="venv/bin/python"
    VENV_PIP="venv/bin/pip"
fi

echo "Installing/checking dependencies..."
$VENV_PIP install -r requirements.txt

echo "========================================================"
echo "[2/2] Running Unit Tests"
echo "========================================================"

$VENV_PYTHON -m pytest tests/test_api_unit.py -v
