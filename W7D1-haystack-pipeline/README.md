# W7D1 - Haystack Pipeline Architecture

## Objective

Implemented a Haystack-based document retrieval system comparing
traditional BM25 retrieval with dense vector retrieval.

## Architecture

PDF Documents
      |
      v
PDF Converter
      |
      v
Document Splitter
      |
      v
InMemoryDocumentStore
      |
      +----------------------+
      |                      |
      v                      v
BM25 Retriever        Dense Embeddings
      |                      |
      v                      v
BM25 Results          Dense Retrieval
      |                      |
      +----------+-----------+
                 |
                 v
          Precision@5
             Comparison

## Documents

Five PDF documents were indexed:

1. NEET Physics 2024
2. NEET Chemistry 2024
3. NEET Biology 2024
4. Bhagavad Gita
5. We Were Never Meant To Be

## Chunking

Documents are split using Haystack's DocumentSplitter.

Configuration:

- Split by: word
- Chunk size: 200 words
- Overlap: 30 words

The pipeline produced 1,771 chunks from the five PDFs.

## Retrieval Methods

### BM25

BM25 performs lexical retrieval using term frequency and inverse
document frequency.

It is effective when query terms directly occur in the documents.

### Dense Retrieval

Dense retrieval converts documents and queries into vector embeddings
using a Sentence Transformers model.

The system uses vector similarity to retrieve semantically related
content.

## Evaluation

The same 50 questions are evaluated using both retrieval methods.

Precision@5 is calculated as:

Precision@5 =
Relevant retrieved documents / 5

Questions are grouped according to their expected source document.

## Output

Generated evaluation files:

- `outputs/bm25_results.txt`
- `outputs/dense_results.txt`
- `outputs/comparison.txt`

## Project Structure

```text
W7D1-haystack-pipeline/
│
├── data/
│   ├── pdfs/
│   │   ├── physics.pdf
│   │   ├── chemistry.pdf
│   │   ├── biology.pdf
│   │   ├── bhagavad_gita.pdf
│   │   └── novel.pdf
│   │
│   └── questions.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── pipeline_bm25.py
│   ├── pipeline_dense.py
│   └── evaluator.py
│
├── scripts/
│   ├── test_bm25.py
│   ├── test_dense.py
│   └── compare_retrieval.py
│
├── outputs/
│   ├── bm25_results.txt
│   ├── dense_results.txt
│   └── comparison.txt
│
├── tests/
│   ├── test_bm25_pipeline.py
│   └── test_dense_pipeline.py
│
├── requirements.txt
├── README.md
├── run.py
└── .gitignore