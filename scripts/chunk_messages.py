import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys
import argparse
import glob
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_MSGS = 60
MAX_HOURS = 72

def parse_ts(ts_str: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError as e:
        logging.error(f"Failed to parse timestamp '{ts_str}': {e}")
        raise # Re-raise to halt processing if timestamps are critical

def main():
    parser = argparse.ArgumentParser(description="Chunk messages from raw_messages.jsonl or conversation-specific files")
    parser.add_argument("--input", type=str, help="Specific input file to chunk (default: merge all raw_messages_*.jsonl files)")
    args = parser.parse_args()

    messages = []
    input_files = []
    
    if args.input:
        # Use specific input file
        inp_path = Path(args.input)
        if not inp_path.exists():
            logging.error(f"Input file not found: {inp_path}")
            sys.exit(1)
        input_files = [inp_path]
        logging.info(f"Chunking messages from: {inp_path}")
    else:
        # Merge all conversation-specific files (raw_messages_*.jsonl) but NOT the master file
        data_dir = Path("data")
        input_files = sorted(data_dir.glob("raw_messages_*.jsonl"))
        if not input_files:
            # Fallback to master file if no conversation-specific files exist
            master_file = data_dir / "raw_messages.jsonl"
            if master_file.exists():
                input_files = [master_file]
                logging.info(f"No conversation-specific files found, using master file: {master_file}")
            else:
                logging.error(f"No input files found in {data_dir}. Please run extract_imessage.py first.")
                sys.exit(1)
        else:
            logging.info(f"Merging {len(input_files)} conversation-specific files for chunking")
    
    try:
        for inp_file in input_files:
            with inp_file.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON on line {line_num + 1} in {inp_file}: {e}. Skipping line.")
                        continue
        logging.info(f"Loaded {len(messages)} raw messages from {len(input_files)} file(s).")
    except IOError as e:
        logging.error(f"Error reading input file(s): {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading messages: {e}")
        sys.exit(1)

    if not messages:
        logging.warning("No messages loaded to chunk. Exiting.")
        sys.exit(0)

    try:
        # The extract script should already be sorting by date, but we sort again just in case
        messages.sort(key=lambda m: (m.get("conversation_id", ""), m.get("timestamp", "")))
    except TypeError as e:
        logging.error(f"Error sorting messages, likely due to missing or malformed keys: {e}")
        sys.exit(1)

    chunks = []
    current_chunk = []

    def flush_chunk():
        if not current_chunk:
            return
            
        lines = []
        for m in current_chunk:
            who = "me" if m.get("direction") == "sent" else "them"
            ts_str_for_format = m.get("timestamp", "")
            try:
                ts_formatted = parse_ts(ts_str_for_format).strftime('%Y-%m-%d %H:%M')
            except Exception:
                ts_formatted = ts_str_for_format # Fallback if parsing fails
                logging.warning(f"Could not format timestamp '{ts_str_for_format}'. Using raw string.")

            # Include message ID for better traceability
            msg_id = m.get('id', '')
            lines.append(f"[{ts_formatted}] ({who}, msg_id:{msg_id}): {m.get('content', '')}")
            
        chunks.append({
            "conversation_id": current_chunk[0].get("conversation_id", "unknown"),
            "start_ts": current_chunk[0].get("timestamp", ""),
            "end_ts": current_chunk[-1].get("timestamp", ""),
            "content": "\n".join(lines),
        })
        current_chunk.clear()

    for m in messages:
        try:
            if not current_chunk:
                current_chunk.append(m)
                continue

            last_message = current_chunk[-1]
            
            is_same_conversation = m.get("conversation_id") == last_message.get("conversation_id")
            
            time_m = parse_ts(m.get("timestamp", ""))
            time_last = parse_ts(last_message.get("timestamp", ""))
            time_difference = time_m - time_last
            
            is_new_chunk = (
                not is_same_conversation or
                len(current_chunk) >= MAX_MSGS or
                time_difference > timedelta(hours=MAX_HOURS)
            )

            if is_new_chunk:
                flush_chunk()

            current_chunk.append(m)
        except Exception as e:
            logging.error(f"Error processing message for chunking (ID: {m.get('id', 'N/A')}): {e}. Skipping message.")
            # Attempt to flush current chunk if it's not empty, to not lose data
            if current_chunk:
                flush_chunk()
            current_chunk.clear() # Clear current chunk to start fresh

    flush_chunk() # Flush the last remaining chunk

    # Determine output file: if input was a conversation-specific file, create conversation-specific chunks
    if args.input:
        # Extract conversation ID from input filename or use a hash
        inp_path = Path(args.input)
        # Try to extract ID from filename like raw_messages_<id>.jsonl
        match = re.search(r'raw_messages_(.+)\.jsonl', inp_path.name)
        if match:
            conv_id = match.group(1)
            OUT = Path(f"data/chunks_{conv_id}.jsonl")
        else:
            # Fallback: use first conversation_id from chunks
            if chunks:
                conv_id_sanitized = re.sub(r'[^\w\-_\.]', '_', chunks[0].get("conversation_id", "unknown")[:50])
                OUT = Path(f"data/chunks_{conv_id_sanitized}.jsonl")
            else:
                OUT = Path("data/chunks.jsonl")
    else:
        # Merging all conversations, use main chunks file
        OUT = Path("data/chunks.jsonl")

    # Ensure output directory exists
    OUT.parent.mkdir(parents=True, exist_ok=True)

    try:
        with OUT.open("w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                chunk["id"] = i # Assigning a simple sequential ID
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logging.info(f"Wrote {len(chunks)} chunks to {OUT}")
    except IOError as e:
        logging.error(f"Error writing to output file '{OUT}': {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while writing chunks: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()