import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
import logging
import sys
import os
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHAT_DB = Path.home() / "Library/Messages/chat.db"
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "raw_messages.jsonl"

APPLE_EPOCH = 978307200  # seconds between 1970 and 2001

def apple_ts_to_iso(ts):
    if ts is None:
        return None
    try:
        # The timestamp is in nanoseconds, convert to seconds
        seconds = ts / 1_000_000_000 + APPLE_EPOCH
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    except (TypeError, ValueError) as e:
        logging.warning(f"Failed to convert timestamp '{ts}' to ISO format: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract iMessages from chat.db")
    parser.add_argument("--phone-numbers", type=str, help="Comma-separated phone numbers to filter (e.g., '+1234567890,+0987654321')")
    parser.add_argument("--all", action="store_true", help="Extract all messages (no filtering)")
    args = parser.parse_args()
    
    logging.info(f"Attempting to extract iMessage data from: {CHAT_DB}")
    
    conn = None
    try:
        conn = sqlite3.connect(CHAT_DB)
        conn.row_factory = sqlite3.Row
        logging.info("Successfully connected to chat.db.")
    except sqlite3.OperationalError as e:
        logging.error(f"Unable to open iMessage database '{CHAT_DB}': {e}. Please ensure you have 'Full Disk Access' enabled for your terminal and that the Messages app is closed.")
        sys.exit(1) # Exit with an error code
    except Exception as e:
        logging.error(f"An unexpected error occurred while connecting to chat.db: {e}")
        sys.exit(1)

    # Build query with optional filtering
    where_clauses = ["m.text IS NOT NULL"]
    
    if not args.all and args.phone_numbers:
        phone_numbers = [p.strip() for p in args.phone_numbers.split(",")]
        phone_filters = " OR ".join([f"c.chat_identifier LIKE '%{pn}%'" for pn in phone_numbers])
        where_clauses.append(f"({phone_filters})")
        logging.info(f"Filtering for phone numbers: {phone_numbers}")
    elif not args.all:
        # Default: extract all messages (no filter)
        logging.info("Extracting all messages (no filter)")
    
    where_clause = " AND ".join(where_clauses)
    
    q = f"""
    SELECT
        m.ROWID as id,
        m.text,
        m.date,
        m.is_from_me,
        c.chat_identifier
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    JOIN chat c ON cmj.chat_id = c.ROWID
    WHERE {where_clause}
    ORDER BY m.date ASC
    """

    messages_extracted = 0
    try:
        with OUT_FILE.open("w", encoding="utf-8") as f:
            cursor = conn.execute(q)
            for row in cursor:
                text = row["text"].strip()
                if not text:
                    continue

                timestamp = apple_ts_to_iso(row["date"])
                if not timestamp:
                    continue

                rec = {
                    "id": row["id"],
                    "timestamp": timestamp,
                    "direction": "sent" if row["is_from_me"] else "received",
                    "conversation_id": row["chat_identifier"],
                    "content": text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                messages_extracted += 1
        logging.info(f"Successfully extracted {messages_extracted} messages to {OUT_FILE}")
    except sqlite3.Error as e:
        logging.error(f"Error executing database query or fetching rows: {e}")
        sys.exit(1)
    except IOError as e:
        logging.error(f"Error writing to output file '{OUT_FILE}': {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during message extraction: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logging.info("Disconnected from chat.db.")

if __name__ == "__main__":
    main()