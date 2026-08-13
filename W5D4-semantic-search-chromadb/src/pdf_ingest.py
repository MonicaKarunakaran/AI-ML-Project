from pathlib import Path
from typing import List, Dict

import chromadb
from pypdf import PdfReader

from src.config import (
    CHROMA_DIR,
    PDF_COLLECTION,
    PDF_PATH,
)
from src.embeddings import EmbeddingModel


def extract_pdf_chunks(
    pdf_path: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Dict]:

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        text = " ".join(text.split())

        start = 0
        chunk_number = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append(
                {
                    "id": (
                        f"page_{page_number}"
                        f"_chunk_{chunk_number}"
                    ),
                    "text": chunk_text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk": chunk_number,
                    },
                }
            )

            chunk_number += 1

            if end >= len(text):
                break

            start = end - chunk_overlap

    return chunks


def ingest_pdf(pdf_path: str | Path = PDF_PATH):

    chunks = extract_pdf_chunks(pdf_path)

    if not chunks:
        raise ValueError(
            "No text could be extracted from the PDF."
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=PDF_COLLECTION,
        metadata={
            "description": "PDF chunks for RAG"
        },
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    embedder = EmbeddingModel()

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.embed_documents(
        documents
    )

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"Successfully ingested "
        f"{len(chunks)} PDF chunks."
    )

    print(
        f"Collection count: "
        f"{collection.count()}"
    )

    return collection


if __name__ == "__main__":
    ingest_pdf()