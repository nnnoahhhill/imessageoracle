# 🔮 The Oracle

A private, personal question‑answering system built on top of your iMessage history.

This project lets you ask questions and get answers grounded in **your real conversations**, not generic web data. It treats messages as memories, looks for patterns over time, and responds reflectively rather than like a search engine.

Its giving personal memory engine

---

## 🚀 Quick Start (Mac)

**For first-time setup:**

```bash
# Clone the repository
git clone https://github.com/yourusername/the-oracle.git
cd the-oracle

# Run the setup script (handles everything!)
./setup.sh

# Start the server
./start.sh
```

Then open your browser to: **http://localhost:8000**

The setup script will:
- ✅ Create a Python virtual environment
- ✅ Install all dependencies
- ✅ Guide you through Full Disk Access setup
- ✅ Extract your iMessages
- ✅ Build search indexes
- ✅ Configure OpenRouter API (optional)

**That's it!** No manual configuration needed.

---

## What this is

- A local ingestion pipeline that turns iMessage history into semantic memory chunks
- A hybrid retriever using vector search + keyword search
- A FastAPI backend that answers questions using those memories
- A Fly.io‑deployable service intended for **single‑user, private use**

This is not a multi‑user SaaS and not meant to ingest data in production.

---

## Architecture overview

```
iMessage chat.db
      ↓
scripts/extract_imessage.py
      ↓
data/raw_messages.jsonl
      ↓
scripts/chunk_messages.py
      ↓
data/chunks.jsonl
      ↓
scripts/build_indexes.py
      ↓
output/ (SQLite + HNSW + Whoosh)
      ↓
FastAPI /chat endpoint
```

Key ideas:
- Messages are chunked into conversational memory units
- Each chunk is embedded once and stored
- Queries retrieve relevant chunks and reason over them

---

## 🤖 How It Works (Simple Explanation)

### The AI Pipeline

1. **Embeddings (Vector Search)**
   - Each message chunk is converted into a "vector" (a list of numbers) that represents its meaning
   - Similar messages have similar vectors (close together in "vector space")
   - When you ask a question, your question is also converted to a vector
   - The system finds message chunks with vectors closest to your question

2. **Keyword Search**
   - Also searches for exact words/phrases from your question
   - Combines with vector search for better results

3. **LLM (Language Model)**
   - Takes the retrieved message chunks and your question
   - Uses GPT-4 (via OpenRouter) to generate an answer
   - The LLM knows today's date and can calculate relative time periods ("yesterday", "last weekend") based on actual message dates

### How to Improve Accuracy

**Better Embeddings:**
- Use a better embedding model (currently `nomic-embed-text`)
- Try OpenAI's `text-embedding-3-large` or Cohere embeddings
- Update `config.yaml` → `embedding.embed_model`

**Better Chunking:**
- Adjust chunk size in `config.yaml` → `chunking.max_messages_per_chunk`
- Smaller chunks = more precise, but more chunks to search
- Larger chunks = more context, but less precise

**Better Retrieval:**
- Increase `retrieval.top_k` to retrieve more chunks (currently 30)
- Enable reranking (`retrieval.use_reranking: true`) - uses LLM to reorder results
- Adjust vector vs keyword weights (`retrieval.weight_vector` and `weight_keyword`)

**Better Prompts:**
- Edit `agent/qa_chain.py` → `prompt_template` to give better instructions
- Add examples of good answers
- Specify the tone/style you want

**More Context:**
- The LLM gets the top 20 chunks by default
- Increase `retrieval.final_top_k` to give it more context (but costs more tokens)

**Date Awareness:**
- The system now knows today's date and uses actual dates from messages
- When asking about time periods, be specific: "what did X do on January 15th?" works better than "what did X do last week?"

---

## Repository structure

```
app.py                  # FastAPI app
agent/
  qa_chain.py           # Prompting + LLM interface
  retriever.py          # Hybrid vector + keyword retrieval
scripts/
  extract_imessage.py   # Extract messages from chat.db
  chunk_messages.py     # Group messages into memory chunks
  build_indexes.py      # Build embeddings and indexes
output/
  messages.sqlite       # Chunk storage
  index.bin             # HNSW vector index
  whoosh/               # Keyword index
```

---

## Requirements

- macOS (for iMessage access)
- Python 3.10+
- Ollama (for local embeddings)
- Fly.io CLI (for deployment)

---

## Step 1: Extract iMessage data

Run locally on your Mac. This script requires read access to your home directory.

```bash
python scripts/extract_imessage.py
```

This reads `~/Library/Messages/chat.db` and produces: `data/raw_messages.jsonl`.

---

## Step 2: Chunk messages into memories

```bash
python scripts/chunk_messages.py
```

This groups messages by conversation and time window into memory chunks.
Output: `data/chunks.jsonl`.

---

## Step 3: Build indexes

This step generates embeddings and builds the search indexes. This can be slow.

```bash
python scripts/build_indexes.py
```

All artifacts are written to the `output/` directory. After this, ingestion is complete.

---

## Step 4: Run locally

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Test the endpoint:
```bash
curl "http://localhost:8000/chat?query=why+do+I+get+burned+out"
```

If responses feel personal and reflective, the system is working.

---

## Step 5: Deployment to Fly.io

This project is designed so **no ingestion happens in production**. The `output/` directory containing the indexes must be deployed with the application.

### Build and deploy

```bash
fly auth login
fly launch
fly volumes create data --size 10 # Adjust size as needed
fly deploy
```

You will need to configure your `fly.toml` to mount the volume at `/app/output`.

---

## API

### GET /health

Returns: `{"ok": true}`

### GET /chat

Query parameter:
- `query`

Response:
```json
{
  "response": "answer text",
  "sources": ["memory snippet", ...]
}
```

---

## Privacy and safety

This project operates on extremely personal data.

- Never expose ingestion endpoints publicly.
- Protect the deployed service with network controls.
- Do not share the dataset or indexes.

Treat this system like a private journal that can talk back.

---

## Deployment to Fly.io

### Prerequisites
- [Fly.io account](https://fly.io/)
- [flyctl installed](https://fly.io/docs/flyctl/install/)
- [OpenRouter API key](https://openrouter.ai/keys)

### Quick Deploy
1. **Set your OpenRouter API key:**
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

2. **Deploy with the script:**
   ```bash
   ./deploy.sh
   ```

### Manual Deploy
1. **Login to Fly.io:**
   ```bash
   flyctl auth login
   ```

2. **Deploy the app:**
   ```bash
   flyctl deploy
   ```

3. **Build indexes locally:**
   ```bash
   source oracle_venv/bin/activate
   python3 scripts/build_indexes.py
   ```

4. **Set secrets:**
   ```bash
   flyctl secrets set OPENROUTER_API_KEY="your-api-key-here"
   ```

5. **Create persistent volume and upload indexes:**
   ```bash
   flyctl volumes create oracle_data --size 1
   ./upload_indexes.sh
   ```

### Testing Your Deployment
```bash
# Get your app URL
flyctl apps list

# Test the API
curl "https://your-app.fly.dev/chat?query=hello"
```

### Configuration
- **LLM Provider:** Set `LLM_PROVIDER` environment variable to `openrouter` (default), `openai`, or `ollama`
- **Memory:** Default 1GB, scale up if needed: `flyctl scale memory 2048`
- **Region:** Default `ord` (Chicago), change in `fly.toml` if needed

### Monitoring
```bash
# View logs
flyctl logs -a the-oracle

# Check status
flyctl status -a the-oracle
```

---

## To-Do List

- [x] **Add Configuration Validation:** Implement a startup check in `app.py` or a separate script to validate `config.yaml` and the existence of required index files.
- [x] **Enhance Error Handling:** Improve error handling in the data pipeline and API to provide more informative messages on failure.
- [x] **Add Basic API Authentication:** Secure the `/chat` endpoint with a simple API key mechanism to protect the deployed application.
- [ ] **Create Unit Tests:** Develop a testing suite (`pytest`) for the data pipeline scripts to ensure they are robust.
