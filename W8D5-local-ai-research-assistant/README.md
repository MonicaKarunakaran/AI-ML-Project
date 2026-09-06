# Local AI Research Assistant

A local AI research assistant that retrieves information from
user-provided documents and generates answers using a local Ollama LLM.

## Objective

The project demonstrates a clean local AI workflow using the approved
AI/ML stack and MLOps practices.

## Features

- Local LLM using Ollama
- Semantic document retrieval
- ChromaDB vector store
- LangGraph workflow orchestration
- MLflow experiment tracking
- Automated unit tests
- Source-aware answers
- No external LLM API required

## Architecture

```text
User Question
      |
      v
LangGraph
      |
      v
Document Retriever
      |
      v
ChromaDB
      |
      v
Relevant Context
      |
      v
Ollama LLM
      |
      v
Research Answer