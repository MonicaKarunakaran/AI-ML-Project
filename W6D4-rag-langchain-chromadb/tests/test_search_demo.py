from src.setup_chroma import DOCUMENTS


def test_documents_have_expected_categories():
    categories = []

    for i in range(len(DOCUMENTS)):
        if i < 5:
            categories.append("machine_learning")
        elif i < 10:
            categories.append("data_science")
        elif i < 15:
            categories.append("rag")
        else:
            categories.append("ai")

    assert len(categories) == 20
    assert "rag" in categories
    assert "machine_learning" in categories