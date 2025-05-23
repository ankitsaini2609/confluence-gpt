# 🧠 Confluence to Chroma LLM Search

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

This project allows you to ingest content from Confluence, store it as vector embeddings using ChromaDB, and query it using a local language model. Ideal for building internal knowledge assistants with private document search capabilities.

## 📁 Repository Structure

- `cacheHuggingFaceModel.py`  
  Downloads and caches a Hugging Face model (e.g., `BAAI/bge-base-en-v1.5`) locally to avoid repeated downloads and enable offline use.

- `ingest_confluence_to_chroma.py`  
  Pulls documents from Confluence, generates embeddings using the cached model, and stores them in ChromaDB for semantic search.

- `query_chroma_with_llm.py`  
  Accepts a natural language question, retrieves relevant documents from ChromaDB, and returns an answer using a local LLM (e.g., via LangChain or Transformers).

## 🧰 Prerequisites

- Python 3.8+
- Confluence access with API token
- Required Python packages (install via `pip install -r requirements.txt`)

## 🚀 Usage
### 1. Ingest Documents from Confluence
Configure your Confluence API keys and space/page identifiers inside the script.
```bash
python ingest_confluence_to_chroma.py
```

### 2. Query the Knowledge Base
```bash
python query_chroma_with_llm.py
```

You’ll be prompted to enter a question. The script will return the most relevant answer based on the indexed content.

## 🔐 Privacy & Security

- No document content is sent to third-party APIs during embedding.
- Currently using local Hugging Face models via cache.


Feel free to ⭐️ this repo if you find it useful!