# W7D5 - Multi-Document RAG System

## Overview

This project implements a Multi-Document Retrieval-Augmented Generation
(RAG) system using LlamaIndex, Ollama, and ChromaDB.

The system can ingest multiple PDF documents, convert their contents into
vector embeddings, store them in ChromaDB, retrieve relevant information,
and generate answers using a local Ollama LLM.

## Architecture

```text
Multiple PDF Documents
          |
          v
   Document Loader
          |
          v
     Text Chunking
          |
          v
   Ollama Embeddings
          |
          v
       ChromaDB
          |
          v
      Retriever
          |
          v
    Relevant Chunks
          |
          v
      Ollama LLM
          |
          v
   Answer + Sources