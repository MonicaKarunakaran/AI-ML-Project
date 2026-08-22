class SimpleVectorStore:
    """
    Lightweight placeholder for document retrieval.

    The actual ChromaDB RAG pipeline was implemented during W5.
    This class keeps the W6 project structure ready for retrieval
    without duplicating the W5 implementation.
    """

    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)

    def similarity_search(self, query: str, k: int = 5):
        return self.documents[:k]