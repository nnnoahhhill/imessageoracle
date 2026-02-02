PYTHON := python3
PIP := pip3
VENV := .venv
ACT := . $(VENV)/bin/activate
CFG := /Users/sidework/fuckin-around/the-oracle/config.yaml
API_PORT ?= 8000

.PHONY: venv install ingest curate index api ui all

venv:
	python3 -m venv $(VENV)
	$(ACT) && $(PIP) install -U pip

install: venv
	$(ACT) && $(PIP) install -r requirements.txt

ingest:
	$(ACT) && $(PYTHON) scripts/ingest.py --config $(CFG)

curate:
	$(ACT) && $(PYTHON) scripts/curate.py --config $(CFG)

index:
	$(ACT) && $(PYTHON) scripts/index.py --config $(CFG)

api:
	$(ACT) && uvicorn api.app:app --host 127.0.0.1 --port $(API_PORT) --reload | cat

ui:
	$(ACT) && streamlit run ui/app.py --server.headless true | cat

all: ingest curate index
