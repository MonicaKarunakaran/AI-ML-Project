from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


def load_documents(pdf_directory: Path):
    """
    Load PDF files, split them into smaller chunks,
    and store the chunks in an in-memory Haystack DocumentStore.
    """

    document_store = InMemoryDocumentStore()

    converter = PyPDFToDocument()

    splitter = DocumentSplitter(
        split_by="word",
        split_length=200,
        split_overlap=30
    )

    writer = DocumentWriter(
        document_store=document_store
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "converter",
        converter
    )

    pipeline.add_component(
        "splitter",
        splitter
    )

    pipeline.add_component(
        "writer",
        writer
    )

    pipeline.connect(
        "converter.documents",
        "splitter.documents"
    )

    pipeline.connect(
        "splitter.documents",
        "writer.documents"
    )

    pdf_files = list(
        pdf_directory.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_directory}"
        )

    print(
        f"Found {len(pdf_files)} PDF documents."
    )

    total_chunks = 0

    for pdf_file in pdf_files:

        print(
            f"Loading: {pdf_file.name}"
        )

        result = pipeline.run(
            {
                "converter": {
                    "sources": [pdf_file]
                }
            }
        )

        chunks_written = result[
            "writer"
        ]["documents_written"]

        total_chunks += chunks_written

        print(
            f"Created {chunks_written} chunks "
            f"from {pdf_file.name}"
        )

    print(
        f"\nTotal chunks in store: "
        f"{document_store.count_documents()}"
    )

    return document_store


if __name__ == "__main__":

    from src.config import PDF_DIR

    store = load_documents(
        PDF_DIR
    )

    print(
        f"\nDocumentStore contains "
        f"{store.count_documents()} chunks."
    )