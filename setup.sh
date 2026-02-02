#!/bin/bash

# The Oracle - Mac Setup Script
# This script sets up The Oracle on a Mac laptop

set -e

echo "🔮 Welcome to The Oracle Setup!"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ This script is designed for macOS only.${NC}"
    exit 1
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed.${NC}"
    echo "Please install Python 3 from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Found Python ${PYTHON_VERSION}${NC}"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ ! -d "oracle_venv" ]; then
    python3 -m venv oracle_venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment
source oracle_venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "📦 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Check for iMessage database
IMESSAGE_DB="$HOME/Library/Messages/chat.db"
echo ""
echo "📱 Checking for iMessage database..."

if [ ! -f "$IMESSAGE_DB" ]; then
    echo -e "${YELLOW}⚠️  iMessage database not found at: $IMESSAGE_DB${NC}"
    echo ""
    echo "This usually means:"
    echo "  1. You need to grant Full Disk Access to Terminal"
    echo "  2. Or you don't have any iMessages"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ Found iMessage database${NC}"
fi

# Check Full Disk Access
echo ""
echo "🔐 Checking Full Disk Access permissions..."
if [ ! -f "$IMESSAGE_DB" ] || [ ! -r "$IMESSAGE_DB" ]; then
    echo -e "${YELLOW}⚠️  Full Disk Access may not be granted${NC}"
    echo ""
    echo "To grant Full Disk Access:"
    echo "  1. Open System Settings > Privacy & Security"
    echo "  2. Click 'Full Disk Access'"
    echo "  3. Add your Terminal app (Terminal, iTerm, etc.)"
    echo "  4. Toggle it ON"
    echo "  5. Restart your terminal completely"
    echo ""
    read -p "Have you granted Full Disk Access? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please grant Full Disk Access and run this script again."
        exit 1
    fi
else
    echo -e "${GREEN}✅ Full Disk Access appears to be granted${NC}"
fi

# Extract iMessages
echo ""
echo "📥 Extracting iMessages..."
read -p "Enter phone numbers to include (comma-separated, or press Enter for all): " PHONE_NUMBERS
echo ""

# Create data directory
mkdir -p data

# Run extraction
if [ -z "$PHONE_NUMBERS" ]; then
    if python3 scripts/extract_imessage.py --all; then
        echo -e "${GREEN}✅ Messages extracted${NC}"
    else
        echo -e "${RED}❌ Failed to extract messages${NC}"
        echo "This might be due to permissions. Make sure Full Disk Access is granted."
        exit 1
    fi
else
    if python3 scripts/extract_imessage.py --phone-numbers "$PHONE_NUMBERS"; then
        echo -e "${GREEN}✅ Messages extracted${NC}"
    else
        echo -e "${RED}❌ Failed to extract messages${NC}"
        echo "This might be due to permissions. Make sure Full Disk Access is granted."
        exit 1
    fi
fi

# Chunk messages
echo ""
echo "📝 Chunking messages..."
if python3 scripts/chunk_messages.py; then
    echo -e "${GREEN}✅ Messages chunked${NC}"
else
    echo -e "${RED}❌ Failed to chunk messages${NC}"
    exit 1
fi

# Build indexes
echo ""
echo "🔨 Building search indexes (this will take a while)..."
read -p "Do you have an OpenRouter API key? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your OpenRouter API key: " OPENROUTER_KEY
    export OPENROUTER_API_KEY="$OPENROUTER_KEY"
else
    echo -e "${YELLOW}⚠️  Building indexes without embeddings (will use mock embeddings)${NC}"
    echo "You can add your OpenRouter API key later and rebuild indexes."
fi

if python3 scripts/build_indexes.py; then
    echo -e "${GREEN}✅ Indexes built${NC}"
else
    echo -e "${YELLOW}⚠️  Index build had issues, but continuing...${NC}"
fi

# Create .env file for API key
if [ ! -z "$OPENROUTER_KEY" ]; then
    echo "OPENROUTER_API_KEY=$OPENROUTER_KEY" > .env
    echo -e "${GREEN}✅ API key saved to .env${NC}"
fi

# Final instructions
echo ""
echo "================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "================================"
echo ""
echo "To start The Oracle:"
echo "  1. Activate the virtual environment:"
echo "     source oracle_venv/bin/activate"
echo ""
echo "  2. Start the server:"
echo "     python3 app.py"
echo ""
echo "  3. Open your browser to:"
echo "     http://localhost:8000"
echo ""
echo "Or run the quick start script:"
echo "     ./start.sh"
echo ""
echo "To add your OpenRouter API key later:"
echo "  1. Get a key from https://openrouter.ai/keys"
echo "  2. Add it to .env file:"
echo "     echo 'OPENROUTER_API_KEY=your-key-here' >> .env"
echo "  3. Rebuild indexes:"
echo "     python3 scripts/build_indexes.py"
echo ""
