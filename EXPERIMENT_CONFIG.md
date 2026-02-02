# 🔬 Experiment Configuration

## Current Prompt

```
Today's date: {today_date}

Answer the question directly using the message history below. Be casual, straightforward, and to the point. No dramatic language or fluff.

IMPORTANT: When answering questions about time periods (like "yesterday", "last weekend", "last week"), use the ACTUAL DATES from the messages. Calculate relative time periods based on the message dates compared to today's date. For example, if today is February 1, 2026 and a message is from January 31, 2026, that's "yesterday". If a message is from January 25-26, 2026, that's "last weekend".

**Messages:**
{context}

**Question:** {query}

**Answer directly:**
```

## Embedding Model

- **Provider**: Ollama
- **Model**: `nomic-embed-text`
- **Dimension**: 768 (default for nomic-embed-text)
- **Prefix**: "Uplifting growth energy: " (currently applied but might not be used)
- **API Endpoint**: `http://localhost:11434/api/embed`

## Retrieval Settings

```yaml
retrieval:
  top_k: 30                    # Initial retrieval (vector + keyword)
  final_top_k: 20             # After reranking
  weight_vector: 0.7          # 70% weight on vector similarity
  weight_keyword: 0.3         # 30% weight on keyword matching
  use_reranking: true         # LLM reranks results for relevance
  expand_query: true          # Expand query with related terms (not implemented)
```

## Chunking Settings

```yaml
chunking:
  max_messages_per_chunk: 60   # Max messages in one chunk
  max_hours_per_chunk: 72      # Max time span (3 days)
```

**Chunking Logic:**
- Messages are grouped by conversation
- New chunk created when:
  - Different conversation
  - Reaches 60 messages
  - Time gap > 72 hours

## LLM Settings

```yaml
curation:
  openrouter_model: "openai/gpt-4o"
  extended_thinking: true
  max_thinking_passes: 3       # Multiple reasoning passes
  use_reranking: true
```

## Data Format

### Raw Message Format (`data/raw_messages.jsonl`)
```json
{
  "id": 437541,
  "timestamp": "2021-11-30T20:26:58.682278+00:00",
  "direction": "received",
  "conversation_id": "brendy!!! (everything lol (+14806207632))",
  "content": "message text here",
  "is_group": false
}
```

### Chunk Format (`data/chunks.jsonl`)
```json
{
  "conversation_id": "brendy!!! (everything lol (+14806207632))",
  "start_ts": "2021-11-30T20:26:58.682278+00:00",
  "end_ts": "2021-12-03T23:24:05.779892+00:00",
  "content": "[2021-11-30 20:26] (them, msg_id:437541): message 1\n[2021-11-30 20:49] (me, msg_id:437489): message 2\n...",
  "id": 0
}
```

### Context Format (sent to LLM)
```
--- Message 1 | Date: January 31, 2026 at 2:30 PM (2026-01-31) | Conversation: brendy!!! ---
[2021-11-30 20:26] (them, msg_id:437541): message content here
[2021-11-30 20:49] (me, msg_id:437489): another message
...

--- Message 2 | Date: February 1, 2026 at 10:15 AM (2026-02-01) | Conversation: brendy!!! ---
...
```

## Current Configuration File

Location: `config.yaml`

Key sections:
- `embedding.embed_model`: "nomic-embed-text"
- `retrieval.top_k`: 30
- `retrieval.final_top_k`: 20
- `chunking.max_messages_per_chunk`: 60
- `chunking.max_hours_per_chunk`: 72

## Experiment Ideas

### 1. Better Embeddings
- Try OpenAI: `text-embedding-3-large` (3072 dim) or `text-embedding-3-small` (1536 dim)
- Try Cohere: `embed-english-v3.0`
- Update `config.yaml` → `embedding.embed_model`

### 2. Chunk Size Tuning
- **Smaller chunks** (20-30 messages): More precise, better for specific questions
- **Larger chunks** (100+ messages): More context, better for broad questions
- Adjust `chunking.max_messages_per_chunk`

### 3. Retrieval Tuning
- Increase `top_k` to 50-100 for more context
- Adjust `weight_vector` vs `weight_keyword` (try 0.5/0.5 or 0.8/0.2)
- Disable reranking to see baseline performance

### 4. Prompt Engineering
- Add examples of good answers
- Specify output format
- Add constraints (e.g., "only use information from messages")
- Edit `agent/qa_chain.py` → `prompt_template`

### 5. Context Window
- Increase `final_top_k` to 30-50 for more context
- Trade-off: More tokens = higher cost but potentially better answers

### 6. Date Formatting
- Currently: "January 31, 2026 at 2:30 PM (2026-01-31)"
- Could add: Day of week, relative time ("3 days ago")
- Edit `agent/qa_chain.py` → context formatting

## Files to Modify

- **Prompt**: `agent/qa_chain.py` (line ~68)
- **Embedding Model**: `config.yaml` (line 49)
- **Retrieval Settings**: `config.yaml` (lines 52-58)
- **Chunking**: `config.yaml` (lines 75-78) and `scripts/chunk_messages.py` (lines 13-14)
- **Context Formatting**: `agent/qa_chain.py` (line ~152)
