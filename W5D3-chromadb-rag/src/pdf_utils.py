from pypdf import PdfReader

def extract_text(pdf_path: str) -> str:

    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[str]:

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks