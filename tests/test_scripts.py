import pytest
import sqlite3
import json
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np # Needed for MockOllamaEmbeddings

# Import the script to be tested
# Need to make sure the sys.path is correct or import directly
import sys
sys.path.append('scripts')
import extract_imessage
import chunk_messages
import build_indexes

# Mock data for chat.db
MOCK_DB_ROWS = [
    (1, "Hello from me!", 1678886400000000000, 1, "chat_abc"),  # 2023-03-15 00:00:00 UTC
    (2, "Hi there!", 1678886460000000000, 0, "chat_abc"),     # 2023-03-15 00:01:00 UTC
    (3, "", 1678886520000000000, 1, "chat_abc"),             # Empty message
    (4, "Another message", 1678886580000000000, 1, "chat_xyz"),# Different conversation
]

@patch('sqlite3.connect')
@patch('pathlib.Path.home')
@patch('builtins.open', new_callable=mock_open)
@patch('extract_imessage.logging') # Mock logging to capture output
def test_extract_imessage_success(mock_logging, mock_builtin_open, mock_path_home, mock_sqlite_connect):
    # Setup mocks
    mock_path_home.return_value = Path('/mock/home')
    mock_db_conn = MagicMock()
    mock_sqlite_connect.return_value = mock_db_conn
    mock_cursor = MagicMock()
    mock_db_conn.execute.return_value = mock_cursor
    
    # Configure cursor to return mock rows
    mock_cursor.__iter__.return_value = iter([sqlite3.Row(row) for row in MOCK_DB_ROWS])
    
    # Run the main function of the script
    extract_imessage.main()

    # Assertions
    mock_path_home.assert_called_once()
    mock_sqlite_connect.assert_called_once_with(Path('/mock/home/Library/Messages/chat.db'))
    mock_db_conn.execute.assert_called_once()
    
    mock_builtin_open.assert_called_once_with(Path('data/raw_messages.jsonl'), 'w', encoding='utf-8')
    mock_file_handle = mock_builtin_open()

    # Expected calls to write
    expected_writes = [
        json.dumps({
            "id": 1,
            "timestamp": "2023-03-15T00:00:00+00:00",
            "direction": "sent",
            "conversation_id": "chat_abc",
            "content": "Hello from me!"
        }, ensure_ascii=False) + "\n",
        json.dumps({
            "id": 2,
            "timestamp": "2023-03-15T00:01:00+00:00",
            "direction": "received",
            "conversation_id": "chat_abc",
            "content": "Hi there!"
        }, ensure_ascii=False) + "\n",
        json.dumps({
            "id": 4,
            "timestamp": "2023-03-15T00:03:00+00:00",
            "direction": "sent",
            "conversation_id": "chat_xyz",
            "content": "Another message"
        }, ensure_ascii=False) + "\n",
    ]
    
    # Check that write was called for each expected message
    mock_file_handle.write.assert_has_calls([
        (w,) for w in expected_writes
    ], any_order=False) # Order should generally be preserved from SQL query ORDER BY m.date ASC

    # Check that error logging was not called for this success case
    mock_logging.error.assert_not_called()
    mock_logging.info.assert_called()

@patch('sqlite3.connect')
@patch('pathlib.Path.home')
@patch('extract_imessage.logging')
def test_extract_imessage_db_error(mock_logging, mock_path_home, mock_sqlite_connect):
    mock_path_home.return_value = Path('/mock/home')
    mock_sqlite_connect.side_effect = sqlite3.OperationalError("Mock DB Error")

    with pytest.raises(SystemExit) as excinfo:
        extract_imessage.main()
    
    assert excinfo.value.code == 1
    mock_logging.error.assert_called_with(
        "Unable to open iMessage database '/mock/home/Library/Messages/chat.db': Mock DB Error. "
        "Please ensure you have 'Full Disk Access' enabled for your terminal and that the Messages app is closed."
    )
    mock_logging.info.assert_called_with("Attempting to extract iMessage data from: /mock/home/Library/Messages/chat.db")

@patch('sqlite3.connect')
@patch('pathlib.Path.home')
@patch('builtins.open', new_callable=mock_open)
@patch('extract_imessage.logging')
def test_extract_imessage_file_write_error(mock_logging, mock_builtin_open, mock_path_home, mock_sqlite_connect):
    mock_path_home.return_value = Path('/mock/home')
    mock_db_conn = MagicMock()
    mock_sqlite_connect.return_value = mock_db_conn
    mock_cursor = MagicMock()
    mock_db_conn.execute.return_value = mock_cursor
    mock_cursor.__iter__.return_value = iter([sqlite3.Row(MOCK_DB_ROWS[0])]) # Single message

    mock_builtin_open.side_effect = IOError("Mock File Write Error")

    with pytest.raises(SystemExit) as excinfo:
        extract_imessage.main()
    
    assert excinfo.value.code == 1
    mock_logging.error.assert_called_with(
        "Error writing to output file 'data/raw_messages.jsonl': Mock File Write Error"
    )

# --- Tests for chunk_messages.py ---
MOCK_RAW_MESSAGES = [
    {"id": 1, "timestamp": "2023-01-01T10:00:00+00:00", "direction": "sent", "conversation_id": "chat_A", "content": "Msg A1"},
    {"id": 2, "timestamp": "2023-01-01T10:01:00+00:00", "direction": "received", "conversation_id": "chat_A", "content": "Msg A2"},
    {"id": 3, "timestamp": "2023-01-01T10:02:00+00:00", "direction": "sent", "conversation_id": "chat_A", "content": "Msg A3"},
    {"id": 4, "timestamp": "2023-01-01T10:03:00+00:00", "direction": "received", "conversation_id": "chat_B", "content": "Msg B1"}, # New conversation
    {"id": 5, "timestamp": "2023-01-03T10:00:00+00:00", "direction": "sent", "conversation_id": "chat_A", "content": "Msg A4 (48h gap)"}, # 48h gap
    {"id": 6, "timestamp": "2023-01-03T10:01:00+00:00", "direction": "received", "conversation_id": "chat_A", "content": "Msg A5"},
]

@patch('builtins.open', new_callable=mock_open)
@patch('chunk_messages.logging')
@patch('pathlib.Path.mkdir') # Mock mkdir to prevent actual directory creation
def test_chunk_messages_success(mock_mkdir, mock_logging, mock_builtin_open):
    # Setup mock for reading input file
    mock_builtin_open.side_effect = [
        mock_open(read_data="".join([json.dumps(m) + "\n" for m in MOCK_RAW_MESSAGES])).return_value,
        mock_open().return_value # For writing output file
    ]
    
    # Run the main function of the script
    chunk_messages.main()

    # Assertions
    # Input file read
    mock_builtin_open.assert_any_call(Path('data/raw_messages.jsonl'), encoding='utf-8')
    # Output file written
    mock_builtin_open.assert_any_call(Path('data/chunks.jsonl'), 'w', encoding='utf-8')

    mock_file_handle_write = mock_builtin_open().write

    # Expected chunks (simplified content for assertion)
    expected_chunk_contents = [
        # First chunk: chat_A messages 1-3
        json.dumps({
            "id": 0,
            "conversation_id": "chat_A",
            "start_ts": "2023-01-01T10:00:00+00:00",
            "end_ts": "2023-01-01T10:02:00+00:00",
            "content": "[2023-01-01 10:00] (me): Msg A1\n[2023-01-01 10:01] (them): Msg A2\n[2023-01-01 10:02] (me): Msg A3"
        }, ensure_ascii=False) + "\n",
        # Second chunk: chat_B message 4 (new conversation)
        json.dumps({
            "id": 1,
            "conversation_id": "chat_B",
            "start_ts": "2023-01-01T10:03:00+00:00",
            "end_ts": "2023-01-01T10:03:00+00:00",
            "content": "[2023-01-01 10:03] (them): Msg B1"
        }, ensure_ascii=False) + "\n",
        # Third chunk: chat_A messages 5-6 (due to 48h gap)
        json.dumps({
            "id": 2,
            "conversation_id": "chat_A",
            "start_ts": "2023-01-03T10:00:00+00:00",
            "end_ts": "2023-01-03T10:01:00+00:00",
            "content": "[2023-01-03 10:00] (me): Msg A4 (48h gap)\n[2023-01-03 10:01] (them): Msg A5"
        }, ensure_ascii=False) + "\n",
    ]
    
    written_content = "".join([call.args[0] for call in mock_file_handle_write.call_args_list])
    
    # Assert that the written content matches expected chunks, potentially ignoring order of chunks
    # We expect 3 chunks, each correctly formatted
    written_chunks = [json.loads(line) for line in written_content.strip().split('\n')]
    assert len(written_chunks) == len(expected_chunk_contents)
    
    # Sort both lists by ID to ensure order for comparison
    written_chunks.sort(key=lambda x: x['id'])
    expected_parsed_chunks = [json.loads(c) for c in expected_chunk_contents]
    expected_parsed_chunks.sort(key=lambda x: x['id'])

    assert written_chunks == expected_parsed_chunks
    
    mock_logging.error.assert_not_called()
    mock_logging.info.assert_called()

@patch('builtins.open', new_callable=mock_open)
@patch('chunk_messages.logging')
@patch('pathlib.Path.mkdir')
def test_chunk_messages_input_file_not_found(mock_mkdir, mock_logging, mock_builtin_open):
    mock_builtin_open.side_effect = FileNotFoundError("Mock File Not Found")
    
    with pytest.raises(SystemExit) as excinfo:
        chunk_messages.main()
    
    assert excinfo.value.code == 1
    mock_logging.error.assert_called_with(
        "Input file not found: data/raw_messages.jsonl. Please run extract_imessage.py first."
    )

@patch('builtins.open', new_callable=mock_open)
@patch('chunk_messages.logging')
@patch('pathlib.Path.mkdir')
def test_chunk_messages_json_decode_error(mock_mkdir, mock_logging, mock_builtin_open):
    invalid_json_data = "{\"id\": 1, \"content\": \"valid\"}\n{\"id\": 2, \"content\": \"invalid\n{\"id\": 3, \"content\": \"valid\"}"
    mock_builtin_open.side_effect = [
        mock_open(read_data=invalid_json_data).return_value,
        mock_open().return_value # For writing output file
    ]

    with pytest.raises(SystemExit) as excinfo: # Exit with error due to no valid messages
        chunk_messages.main()
    
    assert excinfo.value.code == 0 # Should exit with 0 if no chunks are written, but logs error
    mock_logging.error.assert_called_with(
        "Error decoding JSON on line 2 in data/raw_messages.jsonl: Expecting ',' delimiter or '}' at line 1 column 28 (char 27). Skipping line."
    )
    mock_logging.warning.assert_called_with("No messages loaded to chunk. Exiting.")


# --- Tests for build_indexes.py ---
MOCK_CHUNKS_DATA = [
    {"id": 0, "conversation_id": "chat_A", "start_ts": "2023-01-01T10:00:00+00:00", "end_ts": "2023-01-01T10:02:00+00:00", "content": "Chunk A"},
    {"id": 1, "conversation_id": "chat_B", "start_ts": "2023-01-01T10:03:00+00:00", "end_ts": "2023-01-01T10:03:00+00:00", "content": "Chunk B"},
]

# Mocking OllamaEmbeddings
class MockOllamaEmbeddings:
    def __init__(self, model="nomic-embed-text"):
        self.model_name = model
        self.embedding_dim = 768 # Standard for nomic-embed-text

    def embed_query(self, text):
        # Return a deterministic mock embedding for a given text
        seed = sum(ord(c) for c in text)
        np.random.seed(seed)
        return np.random.rand(self.embedding_dim).tolist()

@patch('build_indexes.check_ollama', return_value=True)
@patch('build_indexes.sqlite3.connect')
@patch('build_indexes.hnswlib.Index')
@patch('build_indexes.index.create_in')
@patch('build_indexes.OllamaEmbeddings', side_effect=MockOllamaEmbeddings) # Mock the OllamaEmbeddings class directly
@patch('build_indexes.shutil.rmtree')
@patch('build_indexes.logging')
@patch('builtins.open', new_callable=mock_open)
@patch('pathlib.Path.mkdir') # Mock mkdir to prevent actual directory creation
@patch('build_indexes.Path.exists', return_value=True) # Mock Path.exists for config.yaml and data/chunks.jsonl
@patch('build_indexes.yaml.safe_load', return_value={'embedding': {'embed_model': 'mock-model'}})
def test_build_indexes_success(
    mock_yaml_safe_load, mock_path_exists, mock_mkdir, mock_builtin_open, mock_logging,
    mock_rmtree, MockOllamaEmbeddingsClass, mock_whoosh_create_in, mock_hnswlib_index, mock_sqlite_connect,
    mock_check_ollama
):
    # Setup mocks for file I/O
    mock_builtin_open.side_effect = [
        mock_open(read_data="".join([json.dumps(c) + "\n" for c in MOCK_CHUNKS_DATA])).return_value, # For reading chunks.jsonl
        mock_open().return_value, # For writing HNSW_FILE.meta.json
    ]
    
    # Setup mocks for SQLite
    mock_db_conn = MagicMock()
    mock_sqlite_connect.return_value = mock_db_conn

    # Setup mocks for HNSWLib
    mock_hnsw_instance = MagicMock()
    mock_hnswlib_index.return_value = mock_hnsw_instance

    # Setup mocks for Whoosh
    mock_whoosh_index_instance = MagicMock()
    mock_whoosh_create_in.return_value = mock_whoosh_index_instance
    mock_writer = MagicMock()
    mock_whoosh_index_instance.writer.return_value = mock_writer

    # Run the main function of the script
    build_indexes.main()

    # Assertions for logging
    mock_logging.info.assert_called()
    mock_logging.error.assert_not_called()

    # Assertions for file operations
    mock_path_exists.assert_any_call(Path('config.yaml'))
    mock_path_exists.assert_any_call(Path('data/chunks.jsonl'))
    mock_mkdir.assert_any_call(exist_ok=True) # For output dir
    mock_mkdir.assert_any_call() # For whoosh dir

    # Assertions for SQLite
    mock_sqlite_connect.assert_called_once_with(Path('output/messages.sqlite'))
    mock_db_conn.execute.assert_any_call("DROP TABLE IF EXISTS chunks")
    mock_db_conn.execute.assert_any_call(
        """
        CREATE TABLE chunks (
          id INTEGER PRIMARY KEY,
          conversation_id TEXT,
          start_ts TEXT,
          end_ts TEXT,
          content TEXT
        )
        """
    )
    assert mock_db_conn.execute.call_count > 1 # At least 2 for table create/drop, and inserts
    mock_db_conn.commit.assert_called_once()
    mock_db_conn.close.assert_called_once()

    # Assertions for OllamaEmbeddings
    MockOllamaEmbeddingsClass.assert_called_once_with(model='mock-model') # Check constructor call
    mock_ollama_instance = MockOllamaEmbeddingsClass.return_value
    assert mock_ollama_instance.embed_query.call_count == len(MOCK_CHUNKS_DATA) # Called for each chunk

    # Assertions for HNSWLib
    mock_hnswlib_index.assert_called_once_with(space="cosine", dim=MockOllamaEmbeddings().embedding_dim)
    mock_hnsw_instance.init_index.assert_called_once()
    mock_hnsw_instance.add_items.assert_called_once() # Args are complex, just check call for now
    mock_hnsw_instance.save_index.assert_called_once_with(str(Path('output/index.bin')))
    
    # Assertions for HNSWLib meta.json
    mock_builtin_open.assert_any_call(str(Path('output/index.bin')) + ".meta.json", "w")
    mock_file_handle = mock_builtin_open()
    mock_file_handle.write.assert_called_once() # Check meta.json written

    # Assertions for Whoosh
    mock_rmtree.assert_called_once_with(Path('output/whoosh'))
    mock_whoosh_create_in.assert_called_once() # Args are complex, check call
    mock_whoosh_index_instance.writer.assert_called_once()
    assert mock_writer.add_document.call_count == len(MOCK_CHUNKS_DATA) # Called for each chunk
    mock_writer.commit.assert_called_once()
