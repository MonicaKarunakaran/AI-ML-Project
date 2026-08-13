# W5D4 - Semantic Search with ChromaDB

## Overview

This project demonstrates semantic search using ChromaDB and text embeddings.

The project implements:

- ChromaDB vector storage
- Text embeddings
- 20 sample documents
- Cosine similarity search
- Metadata filtering
- PDF ingestion
- PDF chunking
- Top-3 retrieval
- Ollama-based question answering

## Architecture

User Query
    ↓
Embedding Model
    ↓
Query Embedding
    ↓
ChromaDB
    ↓
Top-K Similar Chunks
    ↓
Context
    ↓
Ollama
    ↓
Grounded Answer

## Technologies

- Python
- ChromaDB
- Sentence Transformers
- Ollama
- PyPDF
- Pytest

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate