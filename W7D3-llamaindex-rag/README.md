# W7D3 — LlamaIndex Document Indexing and RAG

## Objective

Build a document retrieval and question-answering pipeline using LlamaIndex,
Ollama embeddings, and ChromaDB.

## Technologies

- Python
- LlamaIndex
- Ollama
- ChromaDB
- PyTest

## Pipeline

### Basic LlamaIndex

Documents
→ SimpleDirectoryReader
→ Ollama Embeddings
→ VectorStoreIndex
→ QueryEngine
→ Answers

### ChromaDB

Documents
→ LlamaIndex
→ Ollama Embeddings
→ ChromaDB
→ ChromaVectorStore
→ QueryEngine
→ Answers

## Tasks Completed

- Indexed text documents using LlamaIndex.
- Configured Ollama embeddings.
- Created a QueryEngine.
- Executed 10 queries.
- Verified answers against source documents.
- Integrated ChromaDB.
- Re-ran queries using ChromaDB.
- Compared query latency.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt