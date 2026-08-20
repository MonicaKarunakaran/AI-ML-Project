from src.rag_pipeline import build_context, split_documents


def test_split_documents():
    pages = [
        {
            "page": 1,
            "text": (
                "Machine learning is a field of artificial intelligence. "
                "It allows computers to learn patterns from data."
            ),
        }
    ]

    chunks = split_documents(pages)

    assert len(chunks) > 0
    assert chunks[0]["page"] == 1
    assert chunks[0]["text"]


def test_build_context():
    results = {
        "documents": [
            [
                "This is the first retrieved chunk.",
                "This is the second retrieved chunk.",
            ]
        ],
        "metadatas": [
            [
                {"page": 1},
                {"page": 2},
            ]
        ],
    }

    context = build_context(results)

    assert "first retrieved chunk" in context
    assert "second retrieved chunk" in context
    assert "Page 1" in context
    assert "Page 2" in context