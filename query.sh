#!/bin/bash
# Wrapper script for query.py that activates the venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/oracle_venv/bin/activate"
exec python3 "${SCRIPT_DIR}/query.py" "$@"
