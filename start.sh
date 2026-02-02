#!/bin/bash

# Quick start script for The Oracle

set -e

# Activate virtual environment
if [ ! -d "oracle_venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

source oracle_venv/bin/activate

# Load .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  OPENROUTER_API_KEY not set. Using mock LLM responses."
    echo "   Add your key to .env file or export it:"
    echo "   export OPENROUTER_API_KEY='your-key-here'"
    echo ""
fi

# Start the server
echo "🚀 Starting The Oracle..."
echo "📱 Open your browser to: http://localhost:8000"
echo "🛑 Press Ctrl+C to stop"
echo ""

python3 app.py
