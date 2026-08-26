# W6D4 – RAG Pipeline with LangChain + ChromaDB

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using the approved AI/ML stack.

The project combines:

- LangChain
- ChromaDB
- Ollama
- Vector embeddings
- Cosine similarity
- Metadata filtering
- PDF document retrieval

## Architecture

PDF
↓
PDF Text Extraction
↓
Text Chunking
↓
Embeddings
↓
ChromaDB
↓
Similarity Search
↓
Top-3 Relevant Chunks
↓
LangChain Prompt
↓
Ollama LLM
↓
Final Answer

## Project Structure

```text
W6D4-rag-langchain-chromadb/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── sample.pdf
│
├── chroma_db/
│
├── src/
│   ├── __init__.py
│   ├── setup_chroma.py
│   ├── search_demo.py
│   └── rag_pipeline.py
│
└── notebooks/