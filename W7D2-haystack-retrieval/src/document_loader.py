from pathlib import Path

from haystack import Document
from pypdf import PdfReader


def load_documents(data_dir: str = "data") -> list[Document]:
    """Load all PDF files from the data directory."""

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    pdf_files = sorted(data_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_path}")

    documents = []

    for pdf_file in pdf_files:
        reader = PdfReader(str(pdf_file))

        text_parts = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                text_parts.append(text)

        content = "\n".join(text_parts).strip()

        if content:
            documents.append(
                Document(
                    content=content,
                    meta={"source": pdf_file.name}
                )
            )

    if not documents:
        raise ValueError("No readable content found in the PDF files.")

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    for document in documents:
        print(
            f"{document.meta['source']}: "
            f"{len(document.content)} characters"
        )