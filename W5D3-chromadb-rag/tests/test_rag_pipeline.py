from src.pdf_utils import chunk_text

def test_chunk_text():
    text = "A" * 2000

    chunks = chunk_text(
        text,
        chunk_size=500,
        chunk_overlap=50
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)