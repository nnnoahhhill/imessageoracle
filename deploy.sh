#!/bin/bash

# Oracle Deployment Script for Fly.io

set -e

echo "🚀 Deploying The Oracle to Fly.io"

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl not found. Install it from: https://fly.io/docs/flyctl/install/"
    exit 1
fi

# Check if user is logged in
if ! flyctl auth whoami &> /dev/null; then
    echo "❌ Not logged in to Fly.io. Run: flyctl auth login"
    exit 1
fi

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  OPENROUTER_API_KEY not set. Please set it:"
    echo "   export OPENROUTER_API_KEY='your-api-key-here'"
    echo "   Or add it to your fly.toml secrets"
    exit 1
fi

# Build and deploy
echo "📦 Building and deploying..."
flyctl deploy

# Set secrets
echo "🔐 Setting secrets..."
flyctl secrets set OPENROUTER_API_KEY="$OPENROUTER_API_KEY"

# Create volume for data persistence
echo "💾 Creating persistent volume..."
flyctl volumes create oracle_data --size 1

echo "✅ Deployment complete!"
echo ""
echo "🌐 Your Oracle is now live at:"
flyctl apps list | grep the-oracle | awk '{print "   https://" $1 ".fly.dev"}'
echo ""
echo "📚 Test it with:"
echo "   curl 'https://your-app.fly.dev/chat?query=hello'"
echo ""
echo "🔧 Manage your app:"
echo "   flyctl logs -a the-oracle"
echo "   flyctl scale memory 1024 -a the-oracle"