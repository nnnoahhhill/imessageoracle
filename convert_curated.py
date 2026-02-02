#!/usr/bin/env python3
"""
Convert curated.jsonl to raw_messages.jsonl format for testing.
"""

import json
from pathlib import Path
from datetime import datetime

def convert_curated_to_raw():
    """Convert curated message format to raw message format."""
    curated_file = Path("data/curated.jsonl")
    raw_file = Path("data/raw_messages.jsonl")

    if not curated_file.exists():
        print(f"Error: {curated_file} not found")
        return

    raw_file.parent.mkdir(parents=True, exist_ok=True)

    with curated_file.open("r", encoding="utf-8") as f_in, raw_file.open("w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue

            msg = json.loads(line)

            # Convert curated format to raw format
            raw_msg = {
                "id": i,
                "timestamp": f"{msg['day']}T12:00:00.000000+00:00",  # Use noon as default time
                "direction": "sent" if msg["role"] == "self" else "received",
                "conversation_id": msg["sender"] if msg["sender"] != "Me" else "self_conversation",
                "content": msg["content"]
            }

            f_out.write(json.dumps(raw_msg, ensure_ascii=False) + "\n")

    print(f"Converted {i+1} messages from {curated_file} to {raw_file}")

if __name__ == "__main__":
    convert_curated_to_raw()