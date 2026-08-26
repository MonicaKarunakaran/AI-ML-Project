# W7D2 - Haystack Retrieval: BM25 & Dense Retrieval

## Objective

Build and compare BM25 and Dense Retrieval pipelines using Haystack.

## Technologies

- Python
- Haystack 3.1.0
- Sentence Transformers
- Sentence Transformers Haystack Integration
- PyPDF
- PyTest

## Dataset

Five PDF documents were indexed:

1. NEET Physics Question Paper
2. NEET Biology Question Paper
3. NEET Chemistry Question Paper
4. Bhagavad Gita
5. We Were Never Meant to Be

## Project Structure

```text
W7D2-haystack-retrieval/
├── data/
├── src/
│   ├── document_loader.py
│   ├── bm25_pipeline.py
│   ├── dense_pipeline.py
│   └── evaluation.py
├── tests/
├── outputs/
├── README.md
├── self_review.md
├── requirements.txt
└── pytest.ini