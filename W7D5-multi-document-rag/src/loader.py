from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from src.config import DOCUMENTS_DIR


def get_pdf_files():
    """
    Return all PDF files available in the documents directory.
    """
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in: {DOCUMENTS_DIR}"
        )

    return pdf_files


def load_documents():
    """
    Load all PDF documents using LlamaIndex.

    Returns:
        list: LlamaIndex Document objects.
    """
    pdf_files = get_pdf_files()

    print(f"Found {len(pdf_files)} PDF documents.")

    for file in pdf_files:
        print(f"  - {file.name}")

    reader = SimpleDirectoryReader(
        input_dir=str(DOCUMENTS_DIR),
        recursive=False,
        required_exts=[".pdf"],
        filename_as_id=True,
    )

    documents = reader.load_data()

    print(f"\nLoaded {len(documents)} document pages.")

    return documents