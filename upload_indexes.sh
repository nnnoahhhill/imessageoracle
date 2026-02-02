#!/bin/bash

# Upload indexes to Fly.io volume

echo "🚀 Uploading indexes to Fly.io volume"

# Check if volume exists
VOLUME_EXISTS=$(flyctl volumes list -a the-oracle | grep oracle_data || echo "")
if [ -z "$VOLUME_EXISTS" ]; then
    echo "❌ Volume 'oracle_data' not found. Creating it..."
    flyctl volumes create oracle_data --size 1 -a the-oracle
fi

# Check if index files exist locally
if [ ! -f "output/index.bin" ] || [ ! -f "output/messages.sqlite" ]; then
    echo "❌ Index files not found locally. Run 'python3 scripts/build_indexes.py' first."
    exit 1
fi

echo "📁 Uploading index files to volume..."

# Start the app temporarily to mount the volume
flyctl scale count 1 -a the-oracle

# Wait for app to start
sleep 10

# Copy files to the volume
echo "📤 Copying index.bin..."
flyctl ssh sftp -a the-oracle put output/index.bin /app/output/index.bin

echo "📤 Copying index.bin.meta.json..."
flyctl ssh sftp -a the-oracle put output/index.bin.meta.json /app/output/index.bin.meta.json

echo "📤 Copying messages.sqlite..."
flyctl ssh sftp -a the-oracle put output/messages.sqlite /app/output/messages.sqlite

echo "📤 Copying Whoosh index..."
flyctl ssh sftp -a the-oracle put -r output/whoosh /app/output/

echo "✅ Index upload complete!"
echo "🔄 Restarting app to pick up new indexes..."
flyctl apps restart the-oracle

echo "🌐 Your Oracle is now live with real indexes!"
echo "   URL: $(flyctl apps list | grep the-oracle | awk '{print "https://" $1 ".fly.dev"}')"