from fastapi import FastAPI, HTTPException, Form, Request, Depends, Header, status, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import yaml
from pathlib import Path
import logging
import os
import json
import sqlite3
import shutil
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import re
import threading
import uuid
import time
import requests
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agent.retriever import HybridRetriever
from agent.qa_chain import QASystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Job tracking for progress
jobs = {}
jobs_lock = threading.Lock()

# Cache for contacts (refreshes every 5 minutes)
contacts_cache = None
contacts_cache_time = None
contacts_cache_lock = threading.Lock()
CACHE_TTL = 300  # 5 minutes

def update_job(job_id: str, status: str, progress: float, message: str = ""):
	"""Update job progress."""
	with jobs_lock:
		jobs[job_id] = {
			"status": status,  # "running", "completed", "error"
			"progress": progress,  # 0.0 to 1.0
			"message": message,
			"updated_at": datetime.now().isoformat()
		}

def invalidate_contacts_cache():
	"""Invalidate contacts cache to force refresh."""
	global contacts_cache, contacts_cache_time
	with contacts_cache_lock:
		contacts_cache = None
		contacts_cache_time = None

def get_job(job_id: str):
	"""Get job status."""
	with jobs_lock:
		return jobs.get(job_id, {"status": "not_found", "progress": 0.0, "message": "Job not found"})

# Serve static files
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def favicon():
	"""Serve the favicon."""
	favicon_path = Path("frontend/public/favicon.ico")
	if favicon_path.exists():
		return FileResponse(favicon_path)
	# Fallback to static directory if frontend/public doesn't exist
	favicon_static = Path("static/favicon.ico")
	if favicon_static.exists():
		return FileResponse(favicon_static)
	raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/")
async def root():
	"""Serve the main frontend page."""
	index_path = Path("static/index.html")
	if index_path.exists():
		return FileResponse(index_path)
	return {"message": "The Oracle API", "docs": "/docs"}

# --- API Key Authentication (optional, not currently used) ---
# API_KEY is for optional endpoint authentication if needed in the future
# Currently not required - only OPENROUTER_API_KEY is needed for LLM functionality
API_KEY = os.getenv("API_KEY")

# --- Configuration and Index Validation on Startup ---
@app.on_event("startup")
async def validate_config_and_indexes():
    logging.info("Starting application startup validation...")
    
    # 1. Check config.yaml
    config_path = Path("config.yaml")
    if not config_path.exists():
        logging.critical(f"Configuration file not found: {config_path}")
        raise RuntimeError(f"Startup failed: {config_path} not found.")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logging.critical(f"Error loading config.yaml: {e}")
        raise RuntimeError(f"Startup failed: Error loading config.yaml - {e}")

    # Default values from config.yaml in `main`
    output_dir_name = cfg.get("output_dir", "output")
    hnswlib_filename = cfg.get("hnswlib_filename", "index.bin")
    sqlite_filename = cfg.get("sqlite_filename", "messages.sqlite")
    whoosh_dir_name = cfg.get("whoosh_dir_name", "whoosh")

    output_dir = Path(output_dir_name)
    
    # 2. Check output directory
    if not output_dir.exists() or not output_dir.is_dir():
        logging.critical(f"Output directory not found or is not a directory: {output_dir}")
        raise RuntimeError(f"Startup failed: {output_dir} not found or invalid. Please run the data pipeline scripts.")

    # 3. Check for essential index files
    required_files = [
        output_dir / sqlite_filename,
        output_dir / hnswlib_filename,
        output_dir / f"{hnswlib_filename}.meta.json", # HNSWLib also has a meta.json
        output_dir / whoosh_dir_name # Whoosh is a directory
    ]

    for req_path in required_files:
        if not req_path.exists():
            logging.critical(f"Required index component not found: {req_path}")
            raise RuntimeError(f"Startup failed: Missing index component '{req_path}'. Please run the data pipeline scripts.")
        if req_path.is_dir() and req_path.name == whoosh_dir_name and not os.listdir(req_path):
            logging.critical(f"Whoosh index directory is empty: {req_path}")
            raise RuntimeError(f"Startup failed: Whoosh index directory '{req_path}' is empty. Please run the data pipeline scripts.")

    logging.info("Configuration and index validation successful. Application is ready to start.")
    
# Global config object loaded after validation
with open("config.yaml", "r", encoding="utf-8") as f:
	cfg = yaml.safe_load(f)

retr_cfg = cfg.get("retrieval", {})
emb_cfg = cfg.get("embedding", {})

# Initialize retriever and QA system
retriever = HybridRetriever(
	output_dir=cfg.get("output_dir", "output"),
	hnswlib_filename=cfg.get("hnswlib_filename", "index.bin"),
	sqlite_filename=cfg.get("sqlite_filename", "messages.sqlite"),
	whoosh_dir_name=cfg.get("whoosh_dir_name", "whoosh"),
	embedding_model=emb_cfg.get("embed_model"),
	top_k=int(retr_cfg.get("top_k", 30)),
	final_top_k=int(retr_cfg.get("final_top_k", 20)),
	weight_vector=float(retr_cfg.get("weight_vector", 0.7)),
	weight_keyword=float(retr_cfg.get("weight_keyword", 0.3)),
	use_reranking=retr_cfg.get("use_reranking", True),
	expand_query=retr_cfg.get("expand_query", True),
)
qa = QASystem(retriever=retriever, model_config=cfg.get("curation", {}))

# Initialize conversations database
CONVERSATIONS_DB = Path("data/conversations.db")

def init_conversations_db():
	"""Initialize the conversations database."""
	CONVERSATIONS_DB.parent.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(CONVERSATIONS_DB)
	conn.execute("""
		CREATE TABLE IF NOT EXISTS conversations (
			id TEXT PRIMARY KEY,
			title TEXT,
			created_at TEXT,
			updated_at TEXT
		)
	""")
	conn.execute("""
		CREATE TABLE IF NOT EXISTS conversation_messages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			conversation_id TEXT,
			role TEXT,
			content TEXT,
			timestamp TEXT,
			FOREIGN KEY (conversation_id) REFERENCES conversations(id)
		)
	""")
	conn.commit()
	conn.close()

# Initialize on startup
init_conversations_db()


class ChatRequest(BaseModel):
	query: str
	extended_thinking: bool = False
	use_reranking: bool = True


@app.get("/health")
async def health() -> dict:
	return {"ok": True}


@app.get("/ollama/status")
async def ollama_status() -> dict:
	"""Check if Ollama is running and accessible."""
	try:
		response = requests.get("http://localhost:11434/api/tags", timeout=2)
		if response.ok:
			return {
				"running": True,
				"url": "http://localhost:11434",
				"models": response.json().get("models", [])
			}
		else:
			return {
				"running": False,
				"error": f"Ollama returned status {response.status_code}"
			}
	except requests.exceptions.ConnectionError:
		return {
			"running": False,
			"error": "Cannot connect to Ollama. Is it running?",
			"suggestion": "Run 'ollama serve' in a terminal"
		}
	except requests.exceptions.Timeout:
		return {
			"running": False,
			"error": "Ollama connection timed out"
		}
	except Exception as e:
		return {
			"running": False,
			"error": str(e)
		}


@app.post("/ollama/start")
async def ollama_start() -> dict:
	"""Attempt to start Ollama (if not already running)."""
	# First check if it's already running
	status_check = await ollama_status()
	if status_check.get("running"):
		return {
			"success": True,
			"message": "Ollama is already running",
			"status": status_check
		}
	
	# Try to start Ollama in the background
	try:
		# Check if ollama command exists
		result = subprocess.run(
			["which", "ollama"],
			capture_output=True,
			text=True,
			timeout=5
		)
		
		if result.returncode != 0:
			return {
				"success": False,
				"error": "Ollama command not found. Please install Ollama first.",
				"install_url": "https://ollama.ai"
			}
		
		# Try to start ollama serve in background
		# Note: This might not work in all environments (e.g., Fly.io)
		# In production, Ollama should be run as a service
		process = subprocess.Popen(
			["ollama", "serve"],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			start_new_session=True
		)
		
		# Wait a moment to see if it starts
		time.sleep(2)
		
		# Check if it's now running
		status_check = await ollama_status()
		if status_check.get("running"):
			return {
				"success": True,
				"message": "Ollama started successfully",
				"status": status_check
			}
		else:
			return {
				"success": False,
				"error": "Ollama process started but not responding",
				"suggestion": "Check if Ollama is already running or if there's a port conflict"
			}
	except FileNotFoundError:
		return {
			"success": False,
			"error": "Ollama not found. Please install Ollama first.",
			"install_url": "https://ollama.ai"
		}
	except Exception as e:
		logging.error(f"Error starting Ollama: {e}")
		return {
			"success": False,
			"error": f"Failed to start Ollama: {str(e)}",
			"suggestion": "Try running 'ollama serve' manually in a terminal"
		}


@app.get("/chat")
async def chat_get(
	query: str | None = None, 
	conversation_id: str | None = None,
	extended_thinking: bool = False,
	use_reranking: bool = True
) -> dict:
	if not query:
		raise HTTPException(status_code=422, detail="query required")
	
	logging.info(f"Chat query received: '{query}' (conversation_id: {conversation_id})")
	
	try:
		# Filter by conversation if specified - pass to retriever
		logging.info("Calling qa.ask()...")
		res = qa.ask(query, use_reranking=use_reranking, use_extended_thinking=extended_thinking, conversation_id=conversation_id)
		logging.info(f"qa.ask() returned, result type: {type(res.get('result'))}, result length: {len(str(res.get('result', '')))}")
		
		# Check if result is valid
		if not res or "result" not in res:
			logging.error(f"Invalid response from qa.ask(): {res}")
			raise HTTPException(status_code=500, detail="Invalid response from QA system")
		
		result_text = str(res["result"])
		if not result_text or result_text.strip() == "":
			logging.warning("Empty result from qa.ask()")
			result_text = "I couldn't generate a response. Please try again."
		
		# Format sources with condensed previews
		sources = []
		for d in res.get("source_documents", []):
			conv_id = d.get("conversation_id", "unknown")
			start_ts = d.get("start_ts", "")
			full_content = d.get("page_content", "")
			
			# Clean content - remove (them, msg_id:xxx) and (me, msg_id:xxx) patterns
			cleaned_content = re.sub(r'\([^,]+,\s*msg_id:\d+\)\s*', '', full_content)
			
			# Format timestamp nicely
			try:
				if start_ts:
					dt = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
					formatted_date = dt.strftime("%b %d, %Y")
					formatted_time = dt.strftime("%I:%M %p")
					display_date = f"{formatted_date} {formatted_time}"
				else:
					display_date = "Unknown date"
			except:
				display_date = start_ts[:10] if start_ts else "Unknown date"
			
			# Create condensed preview (first 150 chars of cleaned content)
			preview = cleaned_content[:150] + "..." if len(cleaned_content) > 150 else cleaned_content
			
			# Clean conversation ID (remove parentheses and extra info)
			clean_conv_id = conv_id.split('(')[0].strip()
			
			sources.append({
				"conversation": clean_conv_id,
				"date": display_date,
				"preview": preview,
				"full_content": cleaned_content,  # Use cleaned content
				"timestamp": start_ts,
				"conversation_id": conv_id
			})
		
		logging.info(f"Returning response with {len(sources)} sources")
		return {"response": result_text, "sources": sources}
	except HTTPException:
		raise
	except Exception as e:
		logging.error(f"Error processing chat query (GET): {e}", exc_info=True)
		import traceback
		logging.error(f"Traceback: {traceback.format_exc()}")
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/chat")
async def chat_post(
	request: Request,
	query: str | None = None,
	conversation_id: str | None = None,
	form_query: str | None = Form(default=None, alias="query"),
) -> dict:
	final_query = query or form_query
	if not final_query:
		try:
			payload = await request.json()
			if isinstance(payload, dict):
				final_query = payload.get("query")
				conversation_id = conversation_id or payload.get("conversation_id")
		except json.JSONDecodeError: # Catch specific JSON decoding error
			raise HTTPException(status_code=422, detail="Invalid JSON payload")
		except Exception as e:
			logging.error(f"Error parsing request payload: {e}")
			raise HTTPException(status_code=422, detail="Invalid request format")

	if not final_query:
		raise HTTPException(status_code=422, detail="query required")
	
	try:
		# Get extended thinking and reranking flags from payload or query params
		extended = False
		reranking = True
		try:
			payload = await request.json()
			if isinstance(payload, dict):
				extended = payload.get("extended_thinking", False)
				reranking = payload.get("use_reranking", True)
				conversation_id = conversation_id or payload.get("conversation_id")
		except:
			pass
		
		res = qa.ask(final_query, use_reranking=reranking, use_extended_thinking=extended, conversation_id=conversation_id)
		# Format sources with condensed previews
		sources = []
		for d in res.get("source_documents", []):
			conv_id = d.get("conversation_id", "unknown")
			start_ts = d.get("start_ts", "")
			full_content = d.get("page_content", "")
			
			# Clean content - remove (them, msg_id:xxx) and (me, msg_id:xxx) patterns
			cleaned_content = re.sub(r'\([^,]+,\s*msg_id:\d+\)\s*', '', full_content)
			
			# Format timestamp nicely
			try:
				if start_ts:
					dt = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
					formatted_date = dt.strftime("%b %d, %Y")
					formatted_time = dt.strftime("%I:%M %p")
					display_date = f"{formatted_date} {formatted_time}"
				else:
					display_date = "Unknown date"
			except:
				display_date = start_ts[:10] if start_ts else "Unknown date"
			
			# Create condensed preview (first 150 chars of cleaned content)
			preview = cleaned_content[:150] + "..." if len(cleaned_content) > 150 else cleaned_content
			
			# Clean conversation ID (remove parentheses and extra info)
			clean_conv_id = conv_id.split('(')[0].strip()
			
			sources.append({
				"conversation": clean_conv_id,
				"date": display_date,
				"preview": preview,
				"full_content": cleaned_content,  # Use cleaned content
				"timestamp": start_ts,
				"conversation_id": conv_id
			})
		return {"response": res["result"], "sources": sources}
	except Exception as e:
		logging.error(f"Error processing chat query (POST): {e}")
		raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/chat-conversations")
async def list_chat_conversations():
	"""List all saved chat conversations."""
	try:
		conn = sqlite3.connect(CONVERSATIONS_DB)
		conn.row_factory = sqlite3.Row
		cursor = conn.execute(
			"SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
		)
		conversations = []
		for row in cursor:
			conversations.append({
				"id": row["id"],
				"title": row["title"],
				"created_at": row["created_at"],
				"updated_at": row["updated_at"]
			})
		conn.close()
		return {"conversations": conversations}
	except Exception as e:
		logging.error(f"Error listing chat conversations: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat-conversations/{conversation_id}")
async def get_chat_conversation(conversation_id: str):
	"""Get a specific chat conversation with all messages."""
	try:
		conn = sqlite3.connect(CONVERSATIONS_DB)
		conn.row_factory = sqlite3.Row
		
		# Get conversation info
		conv_row = conn.execute(
			"SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
			(conversation_id,)
		).fetchone()
		
		if not conv_row:
			raise HTTPException(status_code=404, detail="Conversation not found")
		
		# Get messages
		cursor = conn.execute(
			"SELECT role, content, timestamp FROM conversation_messages WHERE conversation_id = ? ORDER BY timestamp ASC",
			(conversation_id,)
		)
		messages = []
		for row in cursor:
			messages.append({
				"role": row["role"],
				"content": row["content"],
				"timestamp": row["timestamp"]
			})
		
		conn.close()
		return {
			"id": conv_row["id"],
			"title": conv_row["title"],
			"created_at": conv_row["created_at"],
			"updated_at": conv_row["updated_at"],
			"messages": messages
		}
	except HTTPException:
		raise
	except Exception as e:
		logging.error(f"Error getting chat conversation: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-conversations")
async def save_chat_conversation(request: Request):
	"""Save a new chat conversation or update an existing one."""
	try:
		payload = await request.json()
		conversation_id = payload.get("id")  # Optional - if provided, update existing
		title = payload.get("title", "Untitled Conversation")
		messages = payload.get("messages", [])
		
		if not conversation_id:
			# Create new conversation
			conversation_id = str(uuid.uuid4())
			created_at = datetime.now().isoformat()
		else:
			# Update existing - get created_at
			conn = sqlite3.connect(CONVERSATIONS_DB)
			conn.row_factory = sqlite3.Row
			existing = conn.execute(
				"SELECT created_at FROM conversations WHERE id = ?",
				(conversation_id,)
			).fetchone()
			created_at = existing["created_at"] if existing else datetime.now().isoformat()
			conn.close()
		
		updated_at = datetime.now().isoformat()
		
		conn = sqlite3.connect(CONVERSATIONS_DB)
		
		# Upsert conversation
		conn.execute(
			"""INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at)
			   VALUES (?, ?, ?, ?)""",
			(conversation_id, title, created_at, updated_at)
		)
		
		# Delete old messages and insert new ones
		conn.execute(
			"DELETE FROM conversation_messages WHERE conversation_id = ?",
			(conversation_id,)
		)
		
		for msg in messages:
			conn.execute(
				"""INSERT INTO conversation_messages (conversation_id, role, content, timestamp)
				   VALUES (?, ?, ?, ?)""",
				(conversation_id, msg.get("role", "user"), msg.get("content", ""), msg.get("timestamp", updated_at))
			)
		
		conn.commit()
		conn.close()
		
		return {
			"success": True,
			"id": conversation_id,
			"title": title,
			"created_at": created_at,
			"updated_at": updated_at
		}
	except Exception as e:
		logging.error(f"Error saving chat conversation: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat-conversations/{conversation_id}")
async def delete_chat_conversation(conversation_id: str):
	"""Delete a chat conversation."""
	try:
		conn = sqlite3.connect(CONVERSATIONS_DB)
		conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
		conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
		conn.commit()
		conn.close()
		return {"success": True}
	except Exception as e:
		logging.error(f"Error deleting chat conversation: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_messages(
	q: str = "",
	conversation_id: str = None,
	date_from: str = None,
	date_to: str = None,
	limit: int = 100,
	offset: int = 0,
	sort_order: str = "desc"
):
	"""Search all messages in the database. Returns individual messages parsed from chunks."""
	import re
	from datetime import datetime
	
	try:
		sqlite_path = Path(cfg.get("output_dir", "output")) / cfg.get("sqlite_filename", "messages.sqlite")
		if not sqlite_path.exists():
			return {"results": []}
		
		conn = sqlite3.connect(sqlite_path)
		conn.row_factory = sqlite3.Row
		
		# Build query
		where_clauses = []
		params = []
		
		if q:
			where_clauses.append("content LIKE ?")
			params.append(f"%{q}%")
		
		if conversation_id:
			where_clauses.append("conversation_id = ?")
			params.append(conversation_id)
		
		if date_from:
			where_clauses.append("start_ts >= ?")
			params.append(date_from)
		
		if date_to:
			where_clauses.append("end_ts <= ?")
			params.append(date_to)
		
		where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
		order_sql = "start_ts DESC" if sort_order == "desc" else "start_ts ASC"
		
		# Load chunks in larger batches to parse messages
		# We'll parse messages and then apply offset/limit to the parsed results
		chunk_limit = min(limit * 10, 1000)  # Load more chunks to get enough messages
		query = f"SELECT id, conversation_id, start_ts, end_ts, content FROM chunks WHERE {where_sql} ORDER BY {order_sql} LIMIT ?"
		params.append(chunk_limit)
		
		cursor = conn.execute(query, params)
		
		# Parse chunks into individual messages
		# Pattern: [YYYY-MM-DD HH:MM] (me/them, msg_id:XXX): content
		# Messages are separated by newlines, so split and parse each
		message_pattern = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] \((me|them), msg_id:(\d+)\): (.+)$', re.MULTILINE)
		
		results = []
		for row in cursor:
			content = row["content"]
			conversation_id_val = row["conversation_id"]
			chunk_start_ts = row["start_ts"]
			
			# Find all message matches
			matches = message_pattern.finditer(content)
			for match in matches:
				msg_date, direction, msg_id, msg_content = match.groups()
				msg_content = msg_content.strip()
				
				# Filter out empty messages or OBJ placeholders
				# Check for common placeholder characters: \ufffc (OBJECT REPLACEMENT), \ufffd (REPLACEMENT CHARACTER)
				# Also filter messages that are only whitespace or control characters
				msg_content_clean = msg_content.strip()
				if (not msg_content_clean or 
					'\ufffc' in msg_content or 
					'\ufffd' in msg_content or 
					msg_content_clean.isspace() or
					len(msg_content_clean) == 0):
					continue
				
				# Create full timestamp from date
				try:
					dt = datetime.strptime(msg_date, "%Y-%m-%d %H:%M")
					timestamp = dt.isoformat()
				except:
					timestamp = chunk_start_ts
				
				# Format date for display: Month Day, Year
				try:
					dt = datetime.strptime(msg_date, "%Y-%m-%d %H:%M")
					# Format: "Jan 1, 2025"
					display_date = dt.strftime("%b %d, %Y")
					display_time = dt.strftime("%H:%M")
					# Also create a day key for grouping (YYYY-MM-DD)
					day_key = dt.strftime("%Y-%m-%d")
				except:
					display_date = msg_date.split()[0]
					display_time = msg_date.split()[1] if len(msg_date.split()) > 1 else ""
					day_key = msg_date.split()[0] if msg_date.split() else "Unknown"
				
				# Format without (direction, msg_id:xxx) part
				formatted_msg = f"[{msg_date}] {msg_content}"
				
				# Check if it's a group chat from conversation_id
				is_group = "," in conversation_id_val or ";" in conversation_id_val or "(HTML Import)" in conversation_id_val
				
				results.append({
					"msg_id": int(msg_id),
					"conversation_id": conversation_id_val,
					"timestamp": timestamp,
					"display_date": display_date,
					"display_time": display_time,
					"day_key": day_key,
					"direction": direction,
					"content": msg_content,
					"formatted": formatted_msg,
					"is_group": is_group
				})
		
		conn.close()
		
		# Sort results by timestamp
		results.sort(key=lambda x: x["timestamp"], reverse=(sort_order == "desc"))
		
		# Apply pagination to parsed messages
		total = len(results)
		results = results[offset:offset + limit]
		has_more = (offset + limit) < total
		
		return {
			"results": results,
			"total": total,
			"offset": offset,
			"limit": limit,
			"has_more": has_more
		}
	except Exception as e:
		logging.error(f"Error searching messages: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(conversation_id: str = None):
	"""Get statistics and analytics about messages."""
	import re
	from collections import Counter
	from datetime import datetime, timedelta
	
	try:
		sqlite_path = Path(cfg.get("output_dir", "output")) / cfg.get("sqlite_filename", "messages.sqlite")
		if not sqlite_path.exists():
			return {"error": "Database not found"}
		
		conn = sqlite3.connect(sqlite_path)
		conn.row_factory = sqlite3.Row
		
		# Get all chunks
		where_clause = ""
		params = []
		if conversation_id:
			where_clause = "WHERE conversation_id = ?"
			params.append(conversation_id)
		
		cursor = conn.execute(f"SELECT content, start_ts FROM chunks {where_clause}", params)
		
		message_pattern = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] \((me|them), msg_id:\d+\): (.+)$', re.MULTILINE)
		
		total_sent = 0
		total_received = 0
		all_words = []
		all_dates = []
		all_hours = []
		all_weekdays = []
		total_words = 0
		total_chars = 0
		message_lengths = []
		urls_count = 0
		emojis_count = 0
		questions_count = 0
		exclamations_count = 0
		avg_response_time_hours = []
		last_sent_time = None
		
		for row in cursor:
			content = row["content"]
			matches = message_pattern.finditer(content)
			for match in matches:
				msg_date, direction, msg_content = match.groups()
				msg_content = msg_content.strip()
				
				# Skip empty or OBJ messages (same filtering as search endpoint)
				msg_content_clean = msg_content.strip()
				if (not msg_content_clean or 
					'\ufffc' in msg_content or 
					'\ufffd' in msg_content or 
					msg_content_clean.isspace() or
					len(msg_content_clean) == 0):
					continue
				
				if direction == "me":
					total_sent += 1
					# Track response time (time between received and sent)
					if last_sent_time:
						time_diff = (datetime.strptime(msg_date, "%Y-%m-%d %H:%M") - last_sent_time).total_seconds() / 3600
						if 0 < time_diff < 168:  # Within a week
							avg_response_time_hours.append(time_diff)
					last_sent_time = datetime.strptime(msg_date, "%Y-%m-%d %H:%M")
				else:
					total_received += 1
					last_sent_time = None  # Reset on received message
				
				# Extract words (simple word count)
				words = re.findall(r'\b\w+\b', msg_content.lower())
				all_words.extend(words)
				total_words += len(words)
				total_chars += len(msg_content)
				message_lengths.append(len(msg_content))
				
				# Track dates and times
				try:
					dt = datetime.strptime(msg_date, "%Y-%m-%d %H:%M")
					all_dates.append(dt)
					all_hours.append(dt.hour)
					all_weekdays.append(dt.weekday())  # 0 = Monday, 6 = Sunday
				except:
					pass
				
				# Count URLs
				if re.search(r'https?://', msg_content, re.IGNORECASE):
					urls_count += 1
				
				# Count emojis (simple heuristic)
				emoji_pattern = re.compile(r'[😀-🙏🌀-🗿]|[\U0001F300-\U0001F9FF]', re.UNICODE)
				emojis_count += len(emoji_pattern.findall(msg_content))
				
				# Count questions and exclamations
				if '?' in msg_content:
					questions_count += 1
				if '!' in msg_content:
					exclamations_count += 1
		
		conn.close()
		
		# Calculate stats
		total_messages = total_sent + total_received
		
		# Date range
		if all_dates:
			date_range = (max(all_dates) - min(all_dates)).days + 1
			avg_per_day = total_messages / date_range if date_range > 0 else 0
			
			# Hours in range
			hours_range = date_range * 24
			avg_per_hour = total_messages / hours_range if hours_range > 0 else 0
		else:
			avg_per_day = 0
			avg_per_hour = 0
		
		# Top words
		word_counts = Counter(all_words)
		# Filter out common stop words
		stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
		filtered_words = {word: count for word, count in word_counts.items() if word not in stop_words and len(word) > 2}
		top_words = [{"word": word, "count": count} for word, count in Counter(filtered_words).most_common(3)]
		
		# Time calculations
		# Average typing speed: 40 WPM = 0.67 words/second
		typing_speed_wps = 0.67
		# Average reading speed: 200-250 WPM = ~3.5 words/second
		reading_speed_wps = 3.5
		
		time_typing_seconds = total_words / typing_speed_wps if typing_speed_wps > 0 else 0
		time_reading_seconds = total_words / reading_speed_wps if reading_speed_wps > 0 else 0
		time_conversing_seconds = time_typing_seconds + time_reading_seconds
		
		# Format time
		def format_time(seconds):
			hours = int(seconds // 3600)
			minutes = int((seconds % 3600) // 60)
			if hours > 0:
				return f"{hours}h {minutes}m"
			return f"{minutes}m"
		
		# Calculate additional stats
		avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
		avg_chars_per_message = total_chars / total_messages if total_messages > 0 else 0
		
		# Most active hour
		if all_hours:
			hour_counts = Counter(all_hours)
			most_active_hour = hour_counts.most_common(1)[0][0]
			most_active_hour_count = hour_counts.most_common(1)[0][1]
		else:
			most_active_hour = None
			most_active_hour_count = 0
		
		# Most active day of week
		weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
		if all_weekdays:
			weekday_counts = Counter(all_weekdays)
			most_active_weekday_num = weekday_counts.most_common(1)[0][0]
			most_active_weekday = weekday_names[most_active_weekday_num]
		else:
			most_active_weekday = None
		
		# Average response time
		avg_response_hours = sum(avg_response_time_hours) / len(avg_response_time_hours) if avg_response_time_hours else 0
		
		# Conversation span
		if all_dates:
			first_message = min(all_dates)
			last_message = max(all_dates)
			span_days = (last_message - first_message).days
			span_years = span_days / 365.25
		else:
			span_days = 0
			span_years = 0
			first_message = None
			last_message = None
		
		# Sentiment indicators
		question_ratio = (questions_count / total_messages * 100) if total_messages > 0 else 0
		exclamation_ratio = (exclamations_count / total_messages * 100) if total_messages > 0 else 0
		
		return {
			"total_sent": total_sent,
			"total_received": total_received,
			"total_messages": total_messages,
			"avg_per_day": round(avg_per_day, 2),
			"avg_per_hour": round(avg_per_hour, 4),
			"top_words": top_words,
			"total_words": total_words,
			"total_chars": total_chars,
			"avg_message_length": round(avg_message_length, 1),
			"avg_chars_per_message": round(avg_chars_per_message, 1),
			"time_typing": format_time(time_typing_seconds),
			"time_reading": format_time(time_reading_seconds),
			"time_conversing": format_time(time_conversing_seconds),
			"time_typing_seconds": time_typing_seconds,
			"time_reading_seconds": time_reading_seconds,
			"time_conversing_seconds": time_conversing_seconds,
			"urls_count": urls_count,
			"emojis_count": emojis_count,
			"questions_count": questions_count,
			"exclamations_count": exclamations_count,
			"question_ratio": round(question_ratio, 1),
			"exclamation_ratio": round(exclamation_ratio, 1),
			"most_active_hour": most_active_hour,
			"most_active_hour_count": most_active_hour_count,
			"most_active_weekday": most_active_weekday,
			"avg_response_hours": round(avg_response_hours, 2),
			"span_days": span_days,
			"span_years": round(span_years, 2),
			"first_message": first_message.isoformat() if first_message else None,
			"last_message": last_message.isoformat() if last_message else None
		}
	except Exception as e:
		logging.error(f"Error calculating stats: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/embeddings")
async def get_embeddings(limit: int = 500):
	"""Get message embeddings for 3D visualization."""
	try:
		import numpy as np
		import json
		
		# Load HNSW index metadata
		hnsw_path = Path(cfg.get("output_dir", "output")) / cfg.get("hnswlib_filename", "index.bin")
		meta_path = hnsw_path.with_suffix(hnsw_path.suffix + ".meta.json")
		
		if not meta_path.exists():
			return {"embeddings": [], "messages": []}
		
		with open(meta_path, "r") as f:
			meta = json.load(f)
		
		# Load the index
		import hnswlib
		dim = meta.get("dim", 768)
		index = hnswlib.Index(space='cosine', dim=dim)
		index.load_index(str(hnsw_path))
		
		# Get random sample of vectors (or all if small)
		max_elements = index.element_count
		sample_size = min(limit, max_elements)
		
		# Get embeddings - we'll need to reconstruct from chunks
		sqlite_path = Path(cfg.get("output_dir", "output")) / cfg.get("sqlite_filename", "messages.sqlite")
		conn = sqlite3.connect(sqlite_path)
		conn.row_factory = sqlite3.Row
		
		# Get chunk IDs and content
		cursor = conn.execute("SELECT id, conversation_id, start_ts, content FROM chunks LIMIT ?", (sample_size,))
		
		embeddings = []
		messages = []
		
		# For now, return simplified data - full embedding extraction would require re-embedding
		# This is a placeholder that returns chunk metadata
		for row in cursor:
			messages.append({
				"id": row["id"],
				"conversation_id": row["conversation_id"],
				"timestamp": row["start_ts"],
				"preview": row["content"][:100]
			})
		
		conn.close()
		
		return {
			"embeddings": [],  # Would need to extract from index or re-embed
			"messages": messages,
			"note": "Full embedding extraction requires index access. This endpoint provides message metadata."
		}
	except Exception as e:
		logging.error(f"Error getting embeddings: {e}")
		return {"embeddings": [], "messages": [], "error": str(e)}


def load_contacts_cache():
	"""Load contacts from file and cache them."""
	global contacts_cache, contacts_cache_time
	
	with contacts_cache_lock:
		# Check if cache is still valid
		if contacts_cache is not None and contacts_cache_time is not None:
			age = time.time() - contacts_cache_time
			if age < CACHE_TTL:
				return contacts_cache
		
		logging.info("Loading contacts from raw_messages.jsonl (this may take a moment)...")
		raw_file = Path("data/raw_messages.jsonl")
		if not raw_file.exists():
			return []
		
		contacts = {}
		line_count = 0
		# Sample every Nth line for faster processing on large files
		sample_rate = 1
		file_size = raw_file.stat().st_size
		if file_size > 50_000_000:  # If file > 50MB, sample every 10th line
			sample_rate = 10
		elif file_size > 10_000_000:  # If file > 10MB, sample every 5th line
			sample_rate = 5
		
		with raw_file.open("r", encoding="utf-8") as f:
			for line_num, line in enumerate(f):
				if line_num % sample_rate != 0:
					continue
				line_count += 1
				try:
					msg = json.loads(line)
					conv_id = msg.get("conversation_id", "")
					if not conv_id:
						continue
					
					# Count messages per contact (adjust count for sampling)
					if conv_id not in contacts:
						contacts[conv_id] = {
							"id": conv_id,
							"name": conv_id,
							"message_count": 0,
							"is_group": "," in conv_id or ";" in conv_id or conv_id.startswith("chat")
						}
					contacts[conv_id]["message_count"] += sample_rate  # Adjust for sampling
				except json.JSONDecodeError:
					continue
		
		# Sort by message count (descending)
		contacts_list = sorted(contacts.values(), key=lambda x: x["message_count"], reverse=True)
		
		contacts_cache = contacts_list
		contacts_cache_time = time.time()
		logging.info(f"Loaded {len(contacts_list)} contacts (sampled {line_count} lines)")
		
		return contacts_list


@app.get("/raw-contacts")
async def get_raw_contacts(limit: int = 100, offset: int = 0, search: str = ""):
	"""Get unique list of all contacts/phone numbers from raw_messages.jsonl with pagination."""
	try:
		contacts_list = load_contacts_cache()
		
		# Apply search filter if provided
		if search:
			search_lower = search.lower()
			contacts_list = [c for c in contacts_list if search_lower in c["name"].lower() or search_lower in c["id"].lower()]
		
		# Apply pagination
		total = len(contacts_list)
		contacts_list = contacts_list[offset:offset + limit]
		
		return {
			"contacts": contacts_list,
			"total": total,
			"limit": limit,
			"offset": offset,
			"has_more": offset + limit < total
		}
	except Exception as e:
		logging.error(f"Error reading raw contacts: {e}")
		return {"contacts": [], "error": str(e), "total": 0}


def process_contacts_background(job_id: str, contact_list: list, conversation_name: str):
	"""Background function to process contacts."""
	import subprocess
	import re
	
	try:
		update_job(job_id, "running", 0.1, "Filtering messages...")
		
		raw_file = Path("data/raw_messages.jsonl")
		# Create conversation-specific file instead of overwriting master
		# Sanitize contact ID for filename (remove special chars)
		sanitized_id = re.sub(r'[^\w\-_\.]', '_', contact_list[0] if contact_list else "unknown")
		conversation_file = Path(f"data/raw_messages_{sanitized_id}.jsonl")
		
		# Count total lines first for progress
		total_lines = 0
		if raw_file.exists():
			with raw_file.open("r", encoding="utf-8") as f:
				for _ in f:
					total_lines += 1
		
		# Filter messages to only selected contacts and write to conversation-specific file
		selected_count = 0
		processed_lines = 0
		if raw_file.exists():
			with raw_file.open("r", encoding="utf-8") as infile, \
			     conversation_file.open("w", encoding="utf-8") as outfile:
				for line in infile:
					processed_lines += 1
					if processed_lines % 10000 == 0:
						progress = 0.1 + (processed_lines / total_lines) * 0.2
						update_job(job_id, "running", progress, f"Filtering messages... {processed_lines}/{total_lines}")
					
					try:
						msg = json.loads(line)
						conv_id = msg.get("conversation_id", "")
						if conv_id in contact_list:
							# Update conversation_id with the user's name
							msg["conversation_id"] = f"{conversation_name} ({conv_id})"
							outfile.write(json.dumps(msg, ensure_ascii=False) + "\n")
							selected_count += 1
					except json.JSONDecodeError:
						continue
		
		update_job(job_id, "running", 0.3, f"Saved {selected_count} messages to {conversation_file.name}")
		
		# Run chunking with the conversation-specific file
		update_job(job_id, "running", 0.4, "Chunking messages...")
		logging.info(f"Running chunk_messages.py with input: {conversation_file}")
		try:
			result = subprocess.run(
				["python3", "scripts/chunk_messages.py", "--input", str(conversation_file)],
				capture_output=True,
				text=True,
				timeout=600  # 10 minute timeout
			)
			if result.returncode != 0:
				update_job(job_id, "error", 0.4, f"Chunking failed: {result.stderr}")
				logging.error(f"chunk_messages.py failed: {result.stderr}")
				return
		except subprocess.TimeoutExpired:
			update_job(job_id, "error", 0.4, "Chunking timed out after 10 minutes")
			logging.error("chunk_messages.py timed out")
			return
		
		# Run indexing with progress tracking
		update_job(job_id, "running", 0.6, "Building embeddings and indexes... (this may take several minutes)")
		logging.info("Running build_indexes.py...")
		
		# Use Popen to stream output and update progress
		process = subprocess.Popen(
			["python3", "scripts/build_indexes.py"],
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1,
			universal_newlines=True
		)
		
		# Stream output and update progress using threading
		output_lines = []
		output_lock = threading.Lock()
		start_time = time.time()
		timeout_seconds = 1800  # 30 minute timeout
		
		def read_output():
			"""Read output from subprocess in a separate thread."""
			for line in iter(process.stdout.readline, ''):
				with output_lock:
					output_lines.append(line.strip())
				logging.info(f"build_indexes: {line.strip()}")
		
		# Start output reader thread
		output_thread = threading.Thread(target=read_output, daemon=True)
		output_thread.start()
		
		# Monitor progress while process runs
		progress_base = 0.6
		last_update = start_time
		
		while process.poll() is None:
			# Check for timeout
			elapsed = time.time() - start_time
			if elapsed > timeout_seconds:
				process.kill()
				update_job(job_id, "error", 0.6, f"Indexing timed out after {timeout_seconds/60:.0f} minutes")
				logging.error("build_indexes.py timed out")
				return
			
			# Update progress every 20 seconds
			current_time = time.time()
			if current_time - last_update > 20:
				# Estimate progress based on elapsed time
				# Assume it takes 5-15 minutes depending on data size
				estimated_duration = 600  # 10 minutes default
				progress_estimate = min(0.95, progress_base + (elapsed / estimated_duration) * 0.35)
				
				with output_lock:
					chunk_count = len([l for l in output_lines if 'chunk' in l.lower() or 'embedding' in l.lower()])
				
				update_job(job_id, "running", progress_estimate, 
					f"Building embeddings... ({int(elapsed)}s elapsed, {chunk_count} operations)")
				last_update = current_time
			
			time.sleep(2)  # Check every 2 seconds
		
		# Wait for output thread to finish
		output_thread.join(timeout=5)
		
		# Get final return code
		returncode = process.poll()
		if returncode != 0:
			with output_lock:
				error_output = '\n'.join(output_lines[-20:]) if output_lines else "No output"
			update_job(job_id, "error", 0.6, f"Indexing failed (exit code {returncode}): {error_output}")
			logging.error(f"build_indexes.py failed with exit code {returncode}")
			logging.error(f"Last output: {error_output}")
			return
		
		# Final progress update
		update_job(job_id, "running", 0.95, "Finalizing indexes...")
		
		update_job(job_id, "completed", 1.0, f"Successfully processed {selected_count} messages for '{conversation_name}'")
		# Invalidate cache so contacts refresh
		invalidate_contacts_cache()
		
	except Exception as e:
		logging.error(f"Error processing selected contacts: {e}")
		update_job(job_id, "error", 0.0, str(e))


@app.post("/process-selected-contacts")
async def process_selected_contacts(
	contact_ids: str = Form(...),  # JSON array of selected contact IDs
	conversation_name: str = Form(...)
):
	"""Filter raw_messages.jsonl to only include selected contacts, then chunk and index. Returns job ID for progress tracking."""
	import subprocess
	
	try:
		contact_list = json.loads(contact_ids)
		if not contact_list:
			raise HTTPException(status_code=400, detail="No contacts selected")
		
		# Create job ID
		job_id = str(uuid.uuid4())
		update_job(job_id, "running", 0.0, "Starting processing...")
		
		# Start background thread
		thread = threading.Thread(
			target=process_contacts_background,
			args=(job_id, contact_list, conversation_name),
			daemon=True
		)
		thread.start()
		
		return {
			"success": True,
			"job_id": job_id,
			"message": "Processing started. Use /job-status/{job_id} to check progress."
		}
	except Exception as e:
		logging.error(f"Error starting contact processing: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
	"""Get the status of a background job."""
	return get_job(job_id)


@app.get("/jobs")
async def list_jobs():
	"""List all active/running jobs."""
	with jobs_lock:
		# Return only running jobs and recently completed ones (last 10 minutes)
		active_jobs = {}
		cutoff = datetime.now().timestamp() - 600  # 10 minutes ago
		for job_id, job_data in jobs.items():
			if job_data["status"] == "running" or \
			   (job_data["status"] in ["completed", "error"] and 
			    datetime.fromisoformat(job_data["updated_at"]).timestamp() > cutoff):
				active_jobs[job_id] = job_data
		return {"jobs": active_jobs}


# Cache for available conversations
available_conversations_cache = None
available_conversations_cache_time = None
available_conversations_cache_lock = threading.Lock()

def load_available_conversations_cache():
	"""Load conversations from chat.db and cache them."""
	global available_conversations_cache, available_conversations_cache_time
	
	with available_conversations_cache_lock:
		# Check if cache is still valid
		if available_conversations_cache is not None and available_conversations_cache_time is not None:
			age = time.time() - available_conversations_cache_time
			if age < CACHE_TTL:
				return available_conversations_cache
		
		logging.info("Loading conversations from chat.db...")
		CHAT_DB = Path.home() / "Library/Messages/chat.db"
		
		if not CHAT_DB.exists():
			return []
		
		conn = sqlite3.connect(CHAT_DB)
		conn.row_factory = sqlite3.Row
		
		# Get all unique chat identifiers with message counts
		cursor = conn.execute("""
			SELECT 
				c.chat_identifier,
				COUNT(DISTINCT m.ROWID) as message_count,
				MIN(m.date) as first_message,
				MAX(m.date) as last_message
			FROM chat c
			JOIN chat_message_join cmj ON c.ROWID = cmj.chat_id
			JOIN message m ON cmj.message_id = m.ROWID
			WHERE m.text IS NOT NULL
			GROUP BY c.chat_identifier
			ORDER BY last_message DESC
			LIMIT 500
		""")
		
		conversations = []
		for row in cursor:
			chat_id = row["chat_identifier"]
			# Determine if it's a group chat (has multiple participants)
			is_group = "," in chat_id or ";" in chat_id or chat_id.startswith("chat")
			
			conversations.append({
				"id": chat_id,
				"name": chat_id,
				"message_count": row["message_count"],
				"is_group": is_group,
				"first_message": row["first_message"],
				"last_message": row["last_message"]
			})
		
		conn.close()
		available_conversations_cache = conversations
		available_conversations_cache_time = time.time()
		logging.info(f"Loaded {len(conversations)} conversations")
		
		return conversations


@app.get("/available-conversations")
async def list_available_conversations(limit: int = 100, offset: int = 0, search: str = ""):
	"""List all available conversations from chat.db with pagination."""
	try:
		conversations = load_available_conversations_cache()
		
		# Apply search filter if provided
		if search:
			search_lower = search.lower()
			conversations = [c for c in conversations if search_lower in c["name"].lower() or search_lower in c["id"].lower()]
		
		# Apply pagination
		total = len(conversations)
		conversations = conversations[offset:offset + limit]
		
		return {
			"conversations": conversations,
			"total": total,
			"limit": limit,
			"offset": offset,
			"has_more": offset + limit < total
		}
	except sqlite3.OperationalError as e:
		logging.error(f"Database access error: {e}")
		return {"conversations": [], "error": "Database access denied. Please grant Full Disk Access.", "total": 0}
	except Exception as e:
		logging.error(f"Error listing available conversations: {e}")
		return {"conversations": [], "error": str(e), "total": 0}


@app.post("/extract-conversation")
async def extract_conversation(
	conversation_id: str = Form(...),
	conversation_name: str = Form(...),
	include_group_chats: bool = Form(False)
):
	"""Extract messages for a specific conversation and add them to the database."""
	CHAT_DB = Path.home() / "Library/Messages/chat.db"
	OUT_DIR = Path("data")
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	
	APPLE_EPOCH = 978307200
	
	def apple_ts_to_iso(ts):
		if ts is None:
			return None
		try:
			seconds = ts / 1_000_000_000 + APPLE_EPOCH
			return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
		except:
			return None
	
	try:
		if not CHAT_DB.exists():
			raise HTTPException(status_code=404, detail="chat.db not found")
		
		conn = sqlite3.connect(CHAT_DB)
		conn.row_factory = sqlite3.Row
		
		# Build query to get messages for this conversation
		# Also include group chats if requested
		where_clause = "c.chat_identifier = ?"
		params = [conversation_id]
		
		if include_group_chats:
			# This is simplified - in reality, group chat detection is more complex
			where_clause += " OR c.chat_identifier LIKE ?"
			params.append(f"%{conversation_id}%")
		
		query = f"""
		SELECT
			m.ROWID as id,
			m.text,
			m.date,
			m.is_from_me,
			c.chat_identifier
		FROM message m
		JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
		JOIN chat c ON cmj.chat_id = c.ROWID
		WHERE m.text IS NOT NULL AND ({where_clause})
		ORDER BY m.date ASC
		"""
		
		messages = []
		cursor = conn.execute(query, params)
		
		for row in cursor:
			text = row["text"].strip()
			if not text:
				continue
			
			timestamp = apple_ts_to_iso(row["date"])
			if not timestamp:
				continue
			
			# Determine if it's a group chat (infer from chat_identifier pattern)
			chat_id = row["chat_identifier"]
			is_group = "," in chat_id or ";" in chat_id or chat_id.startswith("chat")
			
			messages.append({
				"id": row["id"],
				"timestamp": timestamp,
				"direction": "sent" if row["is_from_me"] else "received",
				"conversation_id": f"{conversation_name} ({chat_id})",
				"content": text,
				"is_group": bool(is_group),
				"original_chat_id": chat_id
			})
		
		conn.close()
		
		# Append to raw_messages.jsonl (so user can select from all contacts)
		raw_file = OUT_DIR / "raw_messages.jsonl"
		with raw_file.open("a", encoding="utf-8") as f:
			for msg in messages:
				f.write(json.dumps(msg, ensure_ascii=False) + "\n")
		
		# Invalidate cache so contacts refresh
		invalidate_contacts_cache()
		
		return {
			"success": True,
			"messages_extracted": len(messages),
			"conversation_name": conversation_name,
			"message": f"Extracted {len(messages)} messages. They are now available in the contact list above. Select them and click 'Process Selected Contacts' to add them to your indexes."
		}
	except Exception as e:
		logging.error(f"Error extracting conversation: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/import-html-export")
async def import_html_export(
	file: UploadFile = File(...),
	conversation_name: str = Form(...),
	is_group: bool = Form(False)
):
	"""Import messages from an HTML export file."""
	OUT_DIR = Path("data")
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	
	try:
		# Read HTML file
		content = await file.read()
		soup = BeautifulSoup(content, 'html.parser')
		
		messages = []
		message_divs = soup.find_all('div', class_='message')
		
		for msg_div in message_divs:
			# Determine direction
			sent_div = msg_div.find('div', class_='sent')
			received_div = msg_div.find('div', class_='received')
			
			if not sent_div and not received_div:
				continue
			
			direction = "sent" if sent_div else "received"
			msg_container = sent_div if sent_div else received_div
			
			# Extract timestamp
			timestamp_span = msg_container.find('span', class_='timestamp')
			if not timestamp_span:
				continue
			
			timestamp_link = timestamp_span.find('a')
			if timestamp_link:
				timestamp_text = timestamp_link.get_text(strip=True)
			else:
				timestamp_text = timestamp_span.get_text(strip=True)
			
			# Parse timestamp (format: "Sep 02, 2021  6:57:16 PM")
			try:
				dt = datetime.strptime(timestamp_text, "%b %d, %Y  %I:%M:%S %p")
				timestamp = dt.isoformat()
			except:
				try:
					# Try alternative format
					dt = datetime.strptime(timestamp_text, "%b %d, %Y %I:%M:%S %p")
					timestamp = dt.isoformat()
				except:
					logging.warning(f"Could not parse timestamp: {timestamp_text}")
					continue
			
			# Extract sender
			sender_span = msg_container.find('span', class_='sender')
			sender = sender_span.get_text(strip=True) if sender_span else "Unknown"
			
			# Extract message content
			bubble_spans = msg_container.find_all('span', class_='bubble')
			content_parts = []
			for bubble in bubble_spans:
				text = bubble.get_text(strip=True)
				if text and '\ufffc' not in text and '\ufffd' not in text:
					content_parts.append(text)
			
			if not content_parts:
				continue
			
			content = " ".join(content_parts)
			
			# Generate a message ID (use timestamp + hash of content)
			msg_id = hash(f"{timestamp}{content}") % 1000000
			
			messages.append({
				"id": msg_id,
				"timestamp": timestamp,
				"direction": "sent" if sender.lower() in ["me", "you"] else "received",
				"conversation_id": f"{conversation_name} (HTML Import)",
				"content": content,
				"is_group": is_group,
				"original_chat_id": sender if sender != "Me" else "HTML Import"
			})
		
		# Append to raw_messages.jsonl (don't overwrite, so user can select from all contacts)
		raw_file = OUT_DIR / "raw_messages.jsonl"
		with raw_file.open("a", encoding="utf-8") as f:
			for msg in messages:
				f.write(json.dumps(msg, ensure_ascii=False) + "\n")
		
		# Invalidate cache so contacts refresh
		invalidate_contacts_cache()
		
		return {
			"success": True,
			"messages_imported": len(messages),
			"conversation_name": conversation_name,
			"message": f"Imported {len(messages)} messages. They are now available in the contact list above. Select them and click 'Process Selected Contacts' to add them to your indexes."
		}
	except Exception as e:
		logging.error(f"Error importing HTML export: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations")
async def list_conversations():
	"""List all conversations in the database."""
	try:
		sqlite_path = Path(cfg.get("output_dir", "output")) / cfg.get("sqlite_filename", "messages.sqlite")
		if not sqlite_path.exists():
			return {"conversations": []}
		
		conn = sqlite3.connect(sqlite_path)
		conn.row_factory = sqlite3.Row
		
		cursor = conn.execute(
			"SELECT DISTINCT conversation_id FROM chunks ORDER BY conversation_id"
		)
		
		conversations = []
		for row in cursor:
			conversations.append({
				"id": row["conversation_id"],
				"name": row["conversation_id"]
			})
		
		conn.close()
		return {"conversations": conversations}
	except Exception as e:
		logging.error(f"Error listing conversations: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-conversation")
async def upload_conversation(
	file: UploadFile = File(...),
	name: str = Form(...)
):
	"""Upload a new conversation file."""
	try:
		# Create uploads directory
		uploads_dir = Path("data/uploads")
		uploads_dir.mkdir(parents=True, exist_ok=True)
		
		# Save uploaded file
		file_path = uploads_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
		with open(file_path, "wb") as f:
			shutil.copyfileobj(file.file, f)
		
		# Process the file (convert to chunks format if needed)
		# For now, just save it - user can rebuild indexes manually
		return {"message": "File uploaded successfully", "file": str(file_path)}
	except Exception as e:
		logging.error(f"Error uploading conversation: {e}")
		raise HTTPException(status_code=500, detail=str(e))


def rebuild_indexes_background(job_id: str):
	"""Background function to rebuild indexes."""
	import subprocess
	
	try:
		update_job(job_id, "running", 0.1, "Chunking messages...")
		logging.info("Running chunk_messages.py...")
		chunk_result = subprocess.run(
			["python3", "scripts/chunk_messages.py"],
			capture_output=True,
			text=True,
			timeout=300
		)
		if chunk_result.returncode != 0:
			update_job(job_id, "error", 0.1, f"Chunking failed: {chunk_result.stderr}")
			return
		
		update_job(job_id, "running", 0.5, "Building embeddings and indexes... (this may take several minutes)")
		logging.info("Running build_indexes.py...")
		
		# Use Popen to stream output
		process = subprocess.Popen(
			["python3", "scripts/build_indexes.py"],
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1,
			universal_newlines=True
		)
		
		# Stream output and update progress
		output_lines = []
		output_lock = threading.Lock()
		start_time = time.time()
		timeout_seconds = 1800  # 30 minute timeout
		
		def read_output():
			"""Read output from subprocess in a separate thread."""
			for line in iter(process.stdout.readline, ''):
				with output_lock:
					output_lines.append(line.strip())
				logging.info(f"build_indexes: {line.strip()}")
		
		# Start output reader thread
		output_thread = threading.Thread(target=read_output, daemon=True)
		output_thread.start()
		
		# Monitor progress while process runs
		progress_base = 0.5
		last_update = start_time
		
		while process.poll() is None:
			# Check for timeout
			elapsed = time.time() - start_time
			if elapsed > timeout_seconds:
				process.kill()
				update_job(job_id, "error", 0.5, f"Indexing timed out after {timeout_seconds/60:.0f} minutes")
				logging.error("build_indexes.py timed out")
				return
			
			# Update progress every 20 seconds
			current_time = time.time()
			if current_time - last_update > 20:
				estimated_duration = 600  # 10 minutes default
				progress_estimate = min(0.9, progress_base + (elapsed / estimated_duration) * 0.4)
				
				with output_lock:
					chunk_count = len([l for l in output_lines if 'chunk' in l.lower() or 'embedding' in l.lower()])
				
				update_job(job_id, "running", progress_estimate, 
					f"Building embeddings... ({int(elapsed)}s elapsed, {chunk_count} operations)")
				last_update = current_time
			
			time.sleep(2)
		
		# Wait for output thread
		output_thread.join(timeout=5)
		
		# Get return code
		returncode = process.poll()
		
		if returncode == 0:
			# Reload retriever
			update_job(job_id, "running", 0.9, "Reloading indexes...")
			global retriever, qa
			retriever = HybridRetriever(
				output_dir=cfg.get("output_dir", "output"),
				hnswlib_filename=cfg.get("hnswlib_filename", "index.bin"),
				sqlite_filename=cfg.get("sqlite_filename", "messages.sqlite"),
				whoosh_dir_name=cfg.get("whoosh_dir_name", "whoosh"),
				embedding_model=emb_cfg.get("embed_model"),
				top_k=int(retr_cfg.get("top_k", 30)),
				final_top_k=int(retr_cfg.get("final_top_k", 20)),
				weight_vector=float(retr_cfg.get("weight_vector", 0.7)),
				weight_keyword=float(retr_cfg.get("weight_keyword", 0.3)),
				use_reranking=retr_cfg.get("use_reranking", True),
				expand_query=retr_cfg.get("expand_query", True),
			)
			qa = QASystem(retriever=retriever, model_config=cfg.get("curation", {}))
			
			update_job(job_id, "completed", 1.0, "Indexes rebuilt successfully")
		else:
			with output_lock:
				error_output = '\n'.join(output_lines[-20:]) if output_lines else "No output"
			update_job(job_id, "error", 0.5, f"Rebuild failed (exit code {returncode}): {error_output}")
			logging.error(f"build_indexes.py failed with exit code {returncode}")
			logging.error(f"Last output: {error_output}")
	except subprocess.TimeoutExpired:
		update_job(job_id, "error", 0.0, "Index rebuild timed out")
	except Exception as e:
		logging.error(f"Error rebuilding indexes: {e}")
		update_job(job_id, "error", 0.0, str(e))


@app.post("/rebuild-indexes")
async def rebuild_indexes():
	"""Trigger index rebuild. Returns job ID for progress tracking."""
	try:
		job_id = str(uuid.uuid4())
		update_job(job_id, "running", 0.0, "Starting index rebuild...")
		
		# Start background thread
		thread = threading.Thread(
			target=rebuild_indexes_background,
			args=(job_id,),
			daemon=True
		)
		thread.start()
		
		return {
			"success": True,
			"job_id": job_id,
			"message": "Index rebuild started. Use /job-status/{job_id} to check progress."
		}
	except Exception as e:
		logging.error(f"Error starting index rebuild: {e}")
		raise HTTPException(status_code=500, detail=str(e))
