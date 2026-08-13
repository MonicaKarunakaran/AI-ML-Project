#!/bin/bash

echo "Running Exercise 3: RAG with MLflow"

python scripts/run_rag.py \
    data/raw/sample.pdf \
    "Summarize the main concepts discussed in the document."