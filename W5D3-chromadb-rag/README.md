# W5D3 - ChromaDB Vector Store and RAG

## Objective

This project demonstrates vector storage, semantic similarity
search, metadata filtering and Retrieval Augmented Generation
using ChromaDB and Ollama.

## Technologies

- Python
- ChromaDB
- Ollama
- nomic-embed-text
- Llama 3.2
- PyPDF
- MLflow
- pytest

## Exercises

### Exercise 1
Creates a ChromaDB collection containing 20 documents and their
Ollama-generated embeddings.

Demonstrates:
- cosine similarity search
- metadata filtering
- manual result verification

### Exercise 2
Processes a PDF by:
1. Extracting text
2. Splitting text into chunks
3. Creating embeddings
4. Storing chunks in ChromaDB
5. Retrieving top-3 relevant chunks
6. Passing context to Ollama

### Exercise 3

Adds MLflow tracking to the RAG workflow.

## Run

```bash
pip install -r requirements.txt



### W5D3 Self-Review Checklist

## ChromaDB

- ChromaDB installed successfully
- Collection created
- 20 documents added
- Embeddings generated using Ollama
- Cosine similarity search implemented
- Metadata filtering implemented
-  Search results manually verified

## RAG

- PDF added
- PDF text extracted
- Text split into chunks
- Chunk embeddings generated
- Chunks stored in ChromaDB
- Top-3 chunks retrieved
- Retrieved context passed to Ollama
- Generated answer manually verified

## Testing

- Unit tests pass
- RAG pipeline tested
- No unnecessary files committed

## Git

- Correct W5 branch used
- Minimum 2 descriptive commits
- Changes pushed to GitHub
- Pull request created

## CIA

- CIA interaction 1 completed
- CIA interaction 2 completed
- Code review feedback considered