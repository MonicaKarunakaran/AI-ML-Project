# W7D4 – LlamaIndex + Ollama Local RAG

## Overview

This project implements a local Retrieval-Augmented Generation (RAG)
pipeline using LlamaIndex and Ollama.

The project demonstrates how locally hosted Large Language Models (LLMs)
can be used for document question answering without relying on a cloud
LLM API.

## Objectives

- Set up Ollama for local LLM inference.
- Run `llama3.2:3b` locally.
- Use LlamaIndex to build a document index.
- Use Ollama embeddings for local retrieval.
- Implement a custom system prompt.
- Test the RAG pipeline with five questions.
- Compare `llama3.2:3b` and `qwen2.5:3b`.
- Add automated tests using PyTest.

## Technology Stack

- Python
- LlamaIndex
- Ollama
- `llama3.2:3b`
- `qwen2.5:3b`
- `nomic-embed-text`
- PyTest

## Project Structure

```text
W7D4-LocalRAG/
├── data/
│   └── docs/
│       └── sample.txt
├── outputs/
│   └── model_comparison.md
├── scripts/
│   └── run_local_rag.py
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── rag.py
│   └── utils.py
├── tests/
│   └── test_rag.py
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── README.md
└── requirements.txt