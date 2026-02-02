#!/usr/bin/env python3
"""
Query script for the Oracle iMessage QA API
"""
import argparse
import json
import os
import sys
import requests
from typing import Optional

# Default URLs
LOCAL_URL = "http://localhost:8000"
PROD_URL = "https://the-oracle.fly.dev"

def query_api(
    query: str,
    url: str = LOCAL_URL,
    api_key: Optional[str] = None,
    pretty: bool = True,
    show_sources: bool = True,
    max_source_length: int = 200,
    extended_thinking: bool = False,
    use_reranking: bool = True
) -> dict:
    """Query the API and return the response."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    try:
        # Try POST first (better for long queries)
        response = requests.post(
            f"{url}/chat",
            json={
                "query": query,
                "extended_thinking": extended_thinking,
                "use_reranking": use_reranking
            },
            headers=headers,
            timeout=120 if extended_thinking else 60  # Longer timeout for extended thinking
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # Fallback to GET if POST fails
        try:
            params = {
                "query": query,
                "extended_thinking": str(extended_thinking).lower(),
                "use_reranking": str(use_reranking).lower()
            }
            response = requests.get(
                f"{url}/chat",
                params=params,
                headers=headers,
                timeout=120 if extended_thinking else 60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e2:
            print(f"Error: {e2}", file=sys.stderr)
            if hasattr(e2, 'response') and e2.response is not None:
                print(f"Response: {e2.response.text}", file=sys.stderr)
            sys.exit(1)

def format_response(data: dict, pretty: bool, show_sources: bool, max_source_length: int):
    """Format and print the API response."""
    if pretty:
        print("\n" + "="*80)
        print("RESPONSE")
        print("="*80)
        print(data.get("response", ""))
        print("="*80)
        
        if show_sources and "sources" in data:
            print("\n" + "-"*80)
            print("SOURCES")
            print("-"*80)
            for i, source in enumerate(data["sources"], 1):
                preview = source[:max_source_length]
                if len(source) > max_source_length:
                    preview += "..."
                print(f"\n[{i}] {preview}")
            print("-"*80)
    else:
        print(json.dumps(data, indent=2))

def main():
    parser = argparse.ArgumentParser(
        description="Query the Oracle iMessage QA API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query localhost
  python query.py "what did I say about burnout?"
  
  # Query production
  python query.py "what did I say about burnout?" --prod
  
  # Query with custom URL
  python query.py "test query" --url https://custom-api.com
  
  # Raw JSON output
  python query.py "test" --raw
  
  # Hide sources
  python query.py "test" --no-sources
  
  # Extended thinking mode (deeper analysis)
  python query.py "deep question" --extended-thinking
  
  # Disable reranking
  python query.py "test" --no-reranking
        """
    )
    
    parser.add_argument(
        "query",
        help="The question to ask"
    )
    
    parser.add_argument(
        "--url",
        default=None,
        help=f"Custom API URL (default: {LOCAL_URL} for local, {PROD_URL} for --prod)"
    )
    
    parser.add_argument(
        "--prod",
        action="store_true",
        help=f"Use production URL ({PROD_URL})"
    )
    
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication (or set API_KEY env var)"
    )
    
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw JSON instead of formatted response"
    )
    
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source messages"
    )
    
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=sys.maxsize,
        help="Maximum length of source previews (default: unlimited)"
    )
    
    parser.add_argument(
        "--extended-thinking",
        action="store_true",
        help="Enable extended thinking mode (multiple reasoning passes for deeper analysis)"
    )
    
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable reranking of retrieved documents"
    )
    
    args = parser.parse_args()
    
    # Determine URL
    if args.url:
        url = args.url
    elif args.prod:
        url = PROD_URL
    else:
        url = LOCAL_URL
    
    # Get API key
    api_key = args.api_key or os.getenv("API_KEY")
    
    # Query the API
    data = query_api(
        query=args.query,
        url=url,
        api_key=api_key,
        pretty=not args.raw,
        show_sources=not args.no_sources,
        max_source_length=args.max_source_length,
        extended_thinking=args.extended_thinking,
        use_reranking=not args.no_reranking
    )
    
    # Format and print
    format_response(
        data,
        pretty=not args.raw,
        show_sources=not args.no_sources,
        max_source_length=args.max_source_length
    )

if __name__ == "__main__":
    main()
