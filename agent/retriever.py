import json
import os
import sqlite3
from typing import Any, Dict, List
import requests

import numpy as np
import hnswlib
from whoosh.index import open_dir
from whoosh.qparser import QueryParser


class HybridRetriever:
	def __init__(self, output_dir: str, hnswlib_filename: str, sqlite_filename: str, whoosh_dir_name: str,
				  embedding_model: str, top_k: int, weight_vector: float, weight_keyword: float, 
				  final_top_k: int = None, use_reranking: bool = False, expand_query: bool = False) -> None:
		self.output_dir = output_dir
		self.top_k = top_k
		self.final_top_k = final_top_k if final_top_k is not None else top_k
		self.wv = float(weight_vector)
		self.wk = float(weight_keyword)
		self.use_reranking = use_reranking
		self.expand_query = expand_query

		self.hnsw_path = os.path.join(output_dir, hnswlib_filename)
		self.whoosh_path = os.path.join(output_dir, whoosh_dir_name)
		self.sqlite_path = os.path.join(output_dir, sqlite_filename)

		# Check if indexes exist
		if not os.path.exists(self.hnsw_path + ".meta.json"):
			print(f"⚠️  Index files not found at {output_dir}. You need to build indexes first.")
			print("   Run locally: python3 scripts/build_indexes.py")
			print("   Then redeploy or upload indexes to the volume.")
			# Create dummy objects to avoid crashes
			self.index = None
			self.ix = None
			self.embedding_model = embedding_model
			self.ollama_url = "http://localhost:11434/api/embed"
			return

		# Load hnswlib index and meta
		with open(self.hnsw_path + ".meta.json", "r", encoding="utf-8") as mf:
			meta = json.load(mf)
		dim = int(meta.get("dim"))
		self.index = hnswlib.Index(space='cosine', dim=dim)
		self.index.load_index(self.hnsw_path)
		# Increase ef for query time - should be at least top_k * 2, but higher for better recall
		# For large datasets, use higher ef values
		query_ef = max(200, top_k * 4, 100)  # Minimum 200 for better recall on large indexes
		self.index.set_ef(query_ef)

		self.embedding_model = embedding_model
		self.ollama_url = "http://localhost:11434/api/embed"

		self.ix = open_dir(self.whoosh_path)

	def get_embedding(self, text):
		"""Get embedding for text using direct Ollama API call."""
		print(f"   Calling Ollama embedding API: {self.ollama_url}")
		print(f"   Model: {self.embedding_model}, Text length: {len(text)}")
		try:
			response = requests.post(self.ollama_url, json={
				"model": self.embedding_model,
				"input": text
			}, timeout=30)
			print(f"   Ollama response status: {response.status_code}")
			if not response.ok:
				print(f"❌ Embedding failed: {response.status_code} - {response.text}")
				response.raise_for_status()
			embeddings = response.json()["embeddings"][0]
			print(f"   ✅ Embedding received, dimension: {len(embeddings)}")
			return embeddings
		except requests.exceptions.Timeout:
			print(f"❌ Embedding request timed out after 30 seconds")
			raise
		except requests.exceptions.ConnectionError as e:
			print(f"❌ Cannot connect to Ollama at {self.ollama_url}: {e}")
			print(f"   Make sure Ollama is running: ollama serve")
			raise
		except Exception as e:
			print(f"❌ Embedding failed for text: {text[:100]}... Error: {e}")
			import traceback
			traceback.print_exc()
			# Return a mock embedding as fallback
			print("   ⚠️  Using mock embedding as fallback")
			import random
			return [random.random() for _ in range(768)]

	def retrieve(self, query: str, conversation_id: str = None) -> List[Dict[str, Any]]:
		# Check if indexes are loaded
		if self.index is None or self.ix is None:
			print("⚠️  Indexes not loaded, returning error message")
			return [{
				"page_content": "Indexes not available. Please build indexes first by running: python3 scripts/build_indexes.py",
				"metadata": {"id": "error", "conversation_id": "system"}
			}]

		print(f"🔍 Retrieving documents for query: '{query[:50]}...'")
		print(f"   Conversation filter: {conversation_id or 'None'}")
		
		# Vector search
		print("📊 Getting embedding for query...")
		try:
			q_vec = np.asarray([self.get_embedding(query)], dtype=np.float32)
			print(f"✅ Embedding received, shape: {q_vec.shape}")
		except Exception as e:
			print(f"❌ Embedding failed: {e}")
			import traceback
			traceback.print_exc()
			raise
		print("🔎 Running vector search (HNSW)...")
		labels, distances = self.index.knn_query(q_vec, k=self.top_k * 2)
		labels = labels[0].tolist()
		print(f"   Found {len(labels)} vector results")
		# cosine distance -> similarity
		vec_scores = [1.0 - float(d) for d in distances[0].tolist()]
		# Normalize
		if vec_scores:
			v_min, v_max = min(vec_scores), max(vec_scores)
			vec_scores = [(s - v_min) / (v_max - v_min + 1e-9) for s in vec_scores]

		# Keyword search
		print("🔎 Running keyword search (Whoosh)...")
		with self.ix.searcher() as searcher:
			q = QueryParser('content', self.ix.schema).parse(query)
			kw_results = searcher.search(q, limit=self.top_k * 2)
			kw_ids = [int(hit['id']) for hit in kw_results]
			max_score = max([h.score for h in kw_results] or [1.0])
			kw_scores = [float(hit.score) / float(max_score) for hit in kw_results]
		print(f"   Found {len(kw_ids)} keyword results")

		print("🔀 Combining vector and keyword results...")
		combined: Dict[int, float] = {}
		for vid, vscore in zip(labels, vec_scores):
			combined[vid] = combined.get(vid, 0.0) + self.wv * vscore
		for kid, kscore in zip(kw_ids, kw_scores):
			combined[kid] = combined.get(kid, 0.0) + self.wk * kscore

		top_ids = sorted(combined, key=combined.get, reverse=True)[: self.top_k]
		print(f"   Top {len(top_ids)} combined results selected")

		print("📚 Fetching document content from SQLite...")
		conn = sqlite3.connect(self.sqlite_path)
		docs: List[Dict[str, Any]] = []
		for tid in top_ids:
			# Filter by conversation_id if specified
			if conversation_id:
				row = conn.execute(
					'SELECT id, start_ts, end_ts, conversation_id, content FROM chunks WHERE id = ? AND conversation_id = ?',
					(tid, conversation_id)
				).fetchone()
			else:
				row = conn.execute('SELECT id, start_ts, end_ts, conversation_id, content FROM chunks WHERE id = ?', (tid,)).fetchone()
			if row:
				docs.append({
					"id": row[0], 
					"start_ts": row[1], 
					"end_ts": row[2],
					"conversation_id": row[3], 
					"page_content": row[4],
					"metadata": {
						"id": row[0],
						"start_ts": row[1],
						"end_ts": row[2],
						"conversation_id": row[3]
					}
				})
		conn.close()
		print(f"✅ Retrieved {len(docs)} documents")
		return docs
