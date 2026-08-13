#!/bin/bash

echo "Running Exercise 2: PDF RAG"

python -m scripts.run_rag \
    data/raw/sample.pdf \
    "What is the main topic discussed in the document?"