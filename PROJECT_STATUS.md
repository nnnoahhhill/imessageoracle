# Project Status: iMessage Chatbot

This document outlines the current status, architecture, and remaining tasks for the iMessage chatbot project.

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) application that uses your iMessage history to create a chatbot capable of answering questions and providing insights based on your past conversations.

The application is almost fully functional and is well-structured into three main parts:
1.  A **data processing pipeline** to ingest, clean, and index your iMessage data.
2.  A **QA (Question-Answering) agent** that uses the indexed data to answer questions.
3.  Two **user interfaces** to interact with the chatbot (a web API and a Streamlit app).

The entire system runs locally using [Ollama](https://ollama.ai/) for the language model, meaning it does not depend on external APIs (after the initial model download).

## Architecture

The project has the following components:

-   **Configuration (`config.yaml`):** A central file to control all aspects of the application, from file paths and data processing parameters to model names.

-   **Data Pipeline (`scripts/`):**
    1.  `ingest.py`: Reads iMessage conversations exported as HTML files.
    2.  `curate.py`: Filters the messages to find the most relevant and high-quality conversational snippets. This script uses a combination of sentiment analysis and an LLM to score and rank the content.
    3.  `index.py`: Takes the curated data and builds a hybrid search index. This includes a vector index (using HNSWLib) for semantic search and a keyword index (using Whoosh) for specific terms.

-   **QA Agent (`agent/`):**
    1.  `retriever.py`: Implements a `HybridRetriever` that combines the results from both the vector and keyword indexes to find the most relevant message snippets for a given query.
    2.  `qa_chain.py`: Contains the `QASystem` which orchestrates the RAG process. It takes a user's question, uses the `HybridRetriever` to fetch context, and then passes the context and question to an Ollama LLM to generate an answer.

-   **User Interfaces (`api/` and `ui/`):**
    -   `api/app.py`: A FastAPI web server that exposes the `QASystem` as a JSON API endpoint.
    -   `ui/app.py`: A user-friendly web interface built with Streamlit.

    **Important Note:** The Streamlit UI and the FastAPI are two separate, independent ways to run the application. The UI does not call the API; each application loads and runs its own instance of the QA agent.

## Current Status

The project is largely **complete and functional**.

-   [x] The data pipeline (`ingest`, `curate`, `index`) is fully implemented and works as expected.
-   [x] The QA agent and hybrid retriever are fully implemented and functional.
-   [x] Both the FastAPI server and the Streamlit UI are working and can be used to interact with the chatbot.

## How to Run the Application

### 1. Data Processing

You must run the data pipeline first to process your iMessage data.

```bash
# Ingest your exported HTML files
python scripts/ingest.py

# Curate the ingested messages to create high-quality content
python scripts/curate.py

# Index the curated content for retrieval
python scripts/index.py
```

### 2. Run the Chatbot

You can run the chatbot using either the Streamlit UI or the FastAPI server.

**Option A: Streamlit UI (Recommended for users)**

```bash
streamlit run ui/app.py
```

**Option B: FastAPI Server (For programmatic access)**

```bash
uvicorn api.app:app --reload
```

## To-Do List

While the core functionality is complete, the following tasks remain to improve the project's usability and maintainability.

-   [ ] **Create a `README.md` User Guide:** A user-friendly guide is needed to explain:
    -   How to export iMessage data from a Mac.
    -   How to configure `config.yaml`, especially setting up `input_globs` and `me_names`.
    -   The step-by-step process for running the data pipeline.
    -   How to run and interact with the Streamlit UI.

-   [ ] **Refactor Shared Logic:** The `QASystem` initialization code is duplicated in `api/app.py` and `ui/app.py`. This should be moved into a shared factory function (e.g., in `agent/qa_chain.py`) to keep the code DRY (Don't Repeat Yourself).

-   [ ] **Improve Error Handling:** Add more robust error handling and user-friendly error messages, for example, if the CSS selectors in `config.yaml` are incorrect or if an index is missing.

-   [ ] **Finalize Dependencies:** Review and clean up `requirements.txt` to ensure it contains all necessary packages for a fresh installation.
