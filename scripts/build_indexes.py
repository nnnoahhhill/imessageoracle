import json
import sqlite3
import numpy as np
from pathlib import Path
import hnswlib
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from langchain_ollama import OllamaEmbeddings
import shutil
import yaml
import logging
import sys
import os
import requests
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_ollama(model_name):
    """Check if Ollama server is running and the model is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            logging.error(f"Ollama server returned status code {response.status_code}")
            return False
        
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        if model_name not in model_names and f"{model_name}:latest" not in model_names:
            logging.warning(f"Model '{model_name}' not found in Ollama. Ensure it's downloaded.")
        return True
    except Exception as e:
        logging.error(f"Could not connect to Ollama server: {e}")
        return False

def clean_content(text):
    """Clean metadata for better embedding quality."""
    # Remove timestamps like [2018-06-21 06:44]
    text = re.sub(r'[\[]\d{4}-\d{2}-\d{2} \d{2}:\d{2}[\]]', '', text)
    # Remove (me, msg_id:627136): or (them, msg_id:627136):
    text = re.sub(r'\((me|them), msg_id:\d+\):', '', text)
    # Remove object replacement characters
    text = text.replace("\ufffc", "")
    # Truncate very long chunks (Ollama has limits, typically 8192 tokens)
    # Rough estimate: 1 token ≈ 4 chars, so 8192 tokens ≈ 32k chars
    # Be conservative and limit to 20k chars
    max_chars = 20000
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    return text.strip()

def main():
    # --- Configuration Loading ---
    logging.info("Loading configuration from config.yaml...")
    try:
        config_path = Path("config.yaml")
        if not config_path.exists():
            logging.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config.yaml: {e}")
        sys.exit(1)

    emb_cfg = cfg.get("embedding", {})
    ollama_model = emb_cfg.get("embed_model", "nomic-embed-text")

    DATA_FILE = Path("data/chunks.jsonl")
    OUT_DIR = Path("output")
    
    # Ensure output directory exists
    try:
        OUT_DIR.mkdir(exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create output directory {OUT_DIR}: {e}")
        sys.exit(1)

    SQLITE_FILE = OUT_DIR / "messages.sqlite"
    HNSW_FILE = OUT_DIR / "index.bin"
    WHOOSH_DIR = OUT_DIR / "whoosh"

    # --- Ollama Embeddings Initialization ---
    logging.info(f"Initializing embeddings with Ollama model: {ollama_model}")
    
    if not check_ollama(ollama_model):
        logging.error("Ollama server is not reachable. Please start Ollama first.")

    try:
        # Explicitly set base_url
        emb = OllamaEmbeddings(model=ollama_model, base_url="http://localhost:11434")
        # Test embedding
        _ = emb.embed_query("test query") 
        logging.info("Ollama embeddings initialized and tested successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize or connect to Ollama embeddings model '{ollama_model}': {e}")
        sys.exit(1)
        
    # --- Chunks Loading ---
    # Merge all conversation-specific chunk files (chunks_*.jsonl) and main chunks.jsonl if it exists
    data_dir = Path("data")
    chunk_files = sorted(data_dir.glob("chunks_*.jsonl"))
    # Also include main chunks.jsonl if it exists
    main_chunks = data_dir / "chunks.jsonl"
    if main_chunks.exists():
        chunk_files.append(main_chunks)
    
    if not chunk_files:
        logging.error(f"No chunk files found in {data_dir}. Please run chunk_messages.py first.")
        sys.exit(1)
    
    logging.info(f"Loading chunks from {len(chunk_files)} file(s)...")
    chunks = []
    chunk_id_offset = 0
    try:
        for chunk_file in chunk_files:
            file_chunks = []
            with chunk_file.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    try:
                        chunk = json.loads(line)
                        # Reassign IDs to ensure uniqueness when merging
                        chunk["id"] = chunk_id_offset + len(file_chunks)
                        file_chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON on line {line_num + 1} in {chunk_file}: {e}")
                        continue
            chunks.extend(file_chunks)
            chunk_id_offset += len(file_chunks)
        if not chunks:
            logging.warning("No chunks loaded.")
            sys.exit(0)
        logging.info(f"Loaded {len(chunks)} chunks from {len(chunk_files)} file(s).")
    except Exception as e:
        logging.error(f"Error loading chunks: {e}")
        sys.exit(1)

    # --- SQLite Database ---
    logging.info(f"Building SQLite database at: {SQLITE_FILE}")
    conn = None
    try:
if SQLITE_FILE.exists():
    SQLITE_FILE.unlink()
conn = sqlite3.connect(SQLITE_FILE)
conn.execute("""
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  conversation_id TEXT,
  start_ts TEXT,
  end_ts TEXT,
  content TEXT
)
""")
for c in chunks:
    conn.execute(
        "INSERT INTO chunks (id, conversation_id, start_ts, end_ts, content) VALUES (?, ?, ?, ?, ?)",
        (c["id"], c["conversation_id"], c["start_ts"], c["end_ts"], c["content"]),
    )
conn.commit()
        logging.info("SQLite database built successfully.")
    except Exception as e:
        logging.error(f"SQLite database error: {e}")
        sys.exit(1)
    finally:
        if conn:
conn.close()

# --- Embeddings & HNSW Index ---
    logging.info("Generating embeddings for HNSW index...")
vecs = []
    valid_chunk_ids = []
    
    batch_size = 1
    try:
for i, c in enumerate(chunks):
            content = c["content"]
            cleaned = clean_content(content)
            
            if not cleaned:
                logging.debug(f"Skipping empty chunk {c['id']}")
                continue

            logging.info(f"  Embedding chunk {i+1}/{len(chunks)} (ID: {c['id']}): {cleaned[:50]}...")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Add a small delay between requests to avoid overwhelming Ollama
                    if i > 0 and i % 10 == 0:
                        time.sleep(0.1)
                    v = emb.embed_query(cleaned)
                    vecs.append(v)
                    valid_chunk_ids.append(c["id"])
                    break
                except Exception as e:
                    error_msg = str(e)
                    # Check if it's a 500 error or connection issue
                    if "500" in error_msg or "EOF" in error_msg or "Internal Server Error" in error_msg:
                        # Ollama might be overloaded or crashed - wait longer
                        wait_time = 5 * (attempt + 1)
                        if attempt < max_retries - 1:
                            logging.warning(f"Ollama server error for chunk {c['id']} (attempt {attempt+1}): {error_msg[:100]}. Waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logging.error(f"Final failure for chunk {c['id']}: {error_msg[:200]}")
                            logging.warning(f"Skipping chunk {c['id']} after {max_retries} failures. Content preview: {cleaned[:100]}...")
                    else:
                        # Other errors - shorter wait
                        wait_time = 2 * (attempt + 1)
                        if attempt < max_retries - 1:
                            logging.warning(f"Error embedding chunk {c['id']} (attempt {attempt+1}): {error_msg[:100]}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logging.error(f"Final failure for chunk {c['id']}: {error_msg[:200]}")
                            logging.warning(f"Skipping chunk {c['id']} after {max_retries} failures.")

        if not vecs:
            logging.error("No embeddings generated.")
            sys.exit(1)

vecs = np.array(vecs, dtype=np.float32)
dim = vecs.shape[1]
        logging.info(f"Embeddings generated. Dimension: {dim}, Total: {len(vecs)}")
    except Exception as e:
        logging.error(f"Error during embedding generation: {e}")
        sys.exit(1)

    logging.info(f"Building HNSW index at: {HNSW_FILE}")
    try:
index_hnsw = hnswlib.Index(space="cosine", dim=dim)
        # Increase parameters for larger datasets: M controls connections, ef_construction controls search quality during build
        # For large datasets (1000+ items), use higher values
        num_items = len(vecs)
        if num_items > 10000:
            ef_construction = 500
            M = 32
        elif num_items > 5000:
            ef_construction = 400
            M = 24
        elif num_items > 1000:
            ef_construction = 300
            M = 20
        else:
            ef_construction = 200
            M = 16
        logging.info(f"Initializing HNSW index with ef_construction={ef_construction}, M={M} for {num_items} items")
        index_hnsw.init_index(max_elements=num_items, ef_construction=ef_construction, M=M)
        index_hnsw.add_items(vecs, ids=np.array(valid_chunk_ids))
index_hnsw.save_index(str(HNSW_FILE))

with open(str(HNSW_FILE) + ".meta.json", "w") as f:
    json.dump({"dim": dim, "total_items": len(vecs)}, f)
        logging.info("HNSW index built successfully.")
    except Exception as e:
        logging.error(f"Error building HNSW index: {e}")
        sys.exit(1)

# --- Whoosh Index ---
    logging.info(f"Building Whoosh index at: {WHOOSH_DIR}")
    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    try:
if WHOOSH_DIR.exists():
    shutil.rmtree(WHOOSH_DIR)
WHOOSH_DIR.mkdir()

ix = index.create_in(WHOOSH_DIR, schema)
writer = ix.writer()
for c in chunks:
    writer.add_document(id=str(c["id"]), content=c["content"])
writer.commit()
        logging.info("Whoosh index built successfully.")
    except Exception as e:
        logging.error(f"Error building Whoosh index: {e}")
        sys.exit(1)

    logging.info("\nAll indexes have been built in the 'output/' directory.")

if __name__ == "__main__":
    main()
