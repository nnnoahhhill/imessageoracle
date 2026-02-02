import os
from typing import Any, Dict, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .retriever import HybridRetriever


class MockLLM:
    """Mock LLM for testing when Ollama is not available."""

    def __call__(self, messages):
        # Return a simple response based on the query
        query = ""
        for msg in messages:
            if hasattr(msg, 'content'):
                query = msg.content
                break

        return f"This is a mock response to your query: '{query}'. The system is working but using a mock LLM. To get real responses, fix the Ollama configuration."


class QASystem:
    def __init__(self, retriever: HybridRetriever, model_config: Dict[str, Any]) -> None:
        self.retriever = retriever
        
        llm_provider = os.environ.get("LLM_PROVIDER", "openrouter").lower()
        
        if llm_provider == "openai":
            print("Using OpenAI API for LLM.")
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self.llm = ChatOpenAI(
                model=model_config.get("openai_model", "gpt-4-turbo"),
                temperature=0.1,
                openai_api_base="https://openai.openrouter.ai/v1" if os.environ.get("USE_OPENROUTER") else None
            )
        elif llm_provider == "openrouter":
            print("Using OpenRouter for LLM.")
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise ValueError("OPENROUTER_API_KEY environment variable not set.")
            model_name = model_config.get("openrouter_model", "openai/gpt-4o")
            print(f"Using model: {model_name}")
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.1,
                openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                max_tokens=4000,  # Allow longer responses for deep thinking
            )
        else:
            print("Using Ollama model for LLM.")
            self.llm = ChatOllama(model=model_config.get("ollama_model", "llama3"))

        # Extended thinking mode
        self.extended_thinking = model_config.get("extended_thinking", False)
        self.max_thinking_passes = model_config.get("max_thinking_passes", 3)
        
        # Main prompt for direct, casual answers
        self.prompt_template = ChatPromptTemplate.from_template("""
Today's date: {today_date}

Answer the question directly using the message history below. Be casual, straightforward, and to the point. No dramatic language or fluff.

IMPORTANT: When answering questions about time periods (like "yesterday", "last weekend", "last week"), use the ACTUAL DATES from the messages. Calculate relative time periods based on the message dates compared to today's date. For example, if today is February 1, 2026 and a message is from January 31, 2026, that's "yesterday". If a message is from January 25-26, 2026, that's "last weekend".

**Messages:**
{context}

**Question:** {query}

**Answer directly:**
""")
        
        # Reranking prompt
        self.rerank_template = ChatPromptTemplate.from_template("""
Given the user's question and a list of message chunks, rank them by relevance to answering the question.

Question: {query}

Message chunks:
{chunks}

Return a JSON array of chunk indices (0-based) ordered by relevance (most relevant first).
Format: [0, 3, 1, 2, ...]
""")
        
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        # Reranking uses string output, we'll parse JSON from it
        self.rerank_chain = self.rerank_template | self.llm | StrOutputParser()

    def _rerank_docs(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using LLM for better relevance."""
        if len(docs) <= 1:
            return docs
        
        # Format chunks for reranking
        chunks_text = []
        for i, doc in enumerate(docs):
            preview = doc['page_content'][:300]  # Preview for reranking
            chunks_text.append(f"[{i}] {preview}...")
        
        try:
            rerank_result = self.rerank_chain.invoke({
                "query": query,
                "chunks": "\n".join(chunks_text)
            })
            
            # Parse JSON array from response
            import json
            import re
            # Extract JSON array from response
            json_match = re.search(r'\[[\d\s,]+\]', rerank_result)
            if json_match:
                ranked_indices = json.loads(json_match.group())
                # Reorder docs based on ranking
                reranked = [docs[i] for i in ranked_indices if 0 <= i < len(docs)]
                # Add any docs not in ranking
                ranked_set = set(ranked_indices)
                for i, doc in enumerate(docs):
                    if i not in ranked_set:
                        reranked.append(doc)
                return reranked
        except Exception as e:
            print(f"Reranking failed: {e}, using original order")
        
        return docs
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms for better retrieval."""
        # Simple expansion - could be enhanced with LLM
        return query
    
    def ask(self, query: str, use_reranking: bool = True, use_extended_thinking: bool = None, conversation_id: str = None) -> Dict[str, Any]:
        # Expand query if enabled
        expanded_query = self._expand_query(query)
        
        # Retrieve documents (with optional conversation filter)
        docs = self.retriever.retrieve(expanded_query, conversation_id=conversation_id)
        
        # Rerank if enabled
        if use_reranking and len(docs) > 5:
            docs = self._rerank_docs(query, docs)
            # Take top N after reranking
            final_top_k = getattr(self.retriever, 'final_top_k', len(docs))
            docs = docs[:final_top_k]
        
        # Format context with metadata for better referencing
        context_parts = []
        for i, d in enumerate(docs, 1):
            conv_id = d.get("conversation_id", "unknown")
            start_ts = d.get("start_ts", "")
            end_ts = d.get("end_ts", "")
            
            # Format timestamp for readability with full date info
            date_info = ""
            try:
                if start_ts:
                    # Handle both with and without timezone
                    ts_clean = start_ts.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(ts_clean)
                    # Format: "January 31, 2026 at 2:30 PM" for better date understanding
                    date_info = dt.strftime('%B %d, %Y at %I:%M %p')
                    # Also include ISO format for precise date comparison
                    iso_date = dt.strftime('%Y-%m-%d')
                else:
                    date_info = start_ts
                    iso_date = ""
            except Exception:
                date_info = start_ts
                iso_date = ""
            
            # Clean conversation ID
            clean_conv_id = conv_id.split('(')[0].strip() if conv_id else "unknown"
            
            # Include date prominently in context
            context_parts.append(f"--- Message {i} | Date: {date_info} ({iso_date}) | Conversation: {clean_conv_id} ---\n{d['page_content']}")
        
        context = "\n\n".join(context_parts)
        
        # Get today's date for the prompt
        today = datetime.now()
        today_str = today.strftime('%B %d, %Y')
        
        # Extended thinking mode - multiple reasoning passes
        use_extended = use_extended_thinking if use_extended_thinking is not None else self.extended_thinking
        try:
            if use_extended and self.max_thinking_passes > 1:
                # First pass: initial analysis
                print(f"Invoking chain with extended thinking (passes: {self.max_thinking_passes})")
                initial_answer = self.chain.invoke({"context": context, "query": query, "today_date": today_str})
                print(f"Initial answer received, type: {type(initial_answer)}, length: {len(str(initial_answer))}")
                
                # Refinement passes
                refined_answer = initial_answer
                for pass_num in range(1, self.max_thinking_passes):
                    print(f"Refinement pass {pass_num}/{self.max_thinking_passes - 1}")
                    refinement_template = ChatPromptTemplate.from_template("""Today's date: {today_date}

Based on your previous analysis and the context, refine and deepen your answer. When referencing time periods, use ACTUAL DATES from the messages compared to today's date.

Previous analysis:
{previous_answer}

Context:
{context}

Question: {query}

Provide a more nuanced, deeper analysis. Look for patterns you might have missed, consider alternative interpretations, and provide more specific evidence from the messages with actual dates.""")
                    
                    refinement_chain = refinement_template | self.llm | StrOutputParser()
                    refined_answer = refinement_chain.invoke({
                        "previous_answer": refined_answer,
                        "context": context,
                        "query": query,
                        "today_date": today_str
                    })
                    print(f"Refinement pass {pass_num} complete")
                
                answer = refined_answer
            else:
                print(f"Invoking chain (extended: {use_extended}, passes: {self.max_thinking_passes})")
                answer = self.chain.invoke({"context": context, "query": query, "today_date": today_str})
                print(f"Chain invoked, answer type: {type(answer)}, length: {len(str(answer))}")
            
            # Ensure answer is a string
            if answer is None:
                raise ValueError("LLM returned None")
            
            answer_str = str(answer).strip()
            if not answer_str:
                raise ValueError("LLM returned empty string")
            
            print(f"Returning answer: {answer_str[:100]}...")
            return {"result": answer_str, "source_documents": docs}
        except Exception as e:
            import traceback
            print(f"Error in ask() method: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
