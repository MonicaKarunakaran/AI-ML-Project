import chromadb

class ChromaStore:
    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        collection_name: str = "w5d3_documents"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str]
    ):
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 3,
        where: dict | None = None
    ):

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

    def count(self) -> int:
        return self.collection.count()