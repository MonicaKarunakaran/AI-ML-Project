from typing import List, Dict

import chromadb
import ollama

from src.config import (
    CHROMA_DIR,
    PDF_COLLECTION,
    OLLAMA_MODEL,
    TOP_K,
)
from src.embeddings import EmbeddingModel


class RAGPipeline:
    """ChromaDB retrieval + Ollama generation pipeline."""

    def __init__(
        self,
        collection_name: str = PDF_COLLECTION,
        model_name: str = OLLAMA_MODEL,
    ):
        self.model_name = model_name

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

        self.embedder = EmbeddingModel()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Dict]:

        query_embedding = self.embedder.embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved_chunks = []

        for i in range(
            len(results["documents"][0])
        ):
            retrieved_chunks.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return retrieved_chunks

    def generate_answer(
        self,
        question: str,
        chunks: List[Dict],
    ) -> str:

        context_parts = []

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            metadata = chunk["metadata"]

            context_parts.append(
                f"[Source {index} | "
                f"Page {metadata.get('page', 'N/A')}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the
provided context.

If the answer cannot be found in the context,
say that the information is not available
in the provided document.

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using "
                        "retrieved document context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response["message"]["content"]

    def ask(self, question: str):

        chunks = self.retrieve(
            question,
            top_k=TOP_K,
        )

        answer = self.generate_answer(
            question,
            chunks,
        )

        return answer, chunks


if __name__ == "__main__":

    pipeline = RAGPipeline()

    question = input(
        "Enter your question: "
    )

    answer, chunks = pipeline.ask(question)

    print("\n" + "=" * 60)
    print("TOP-3 RETRIEVED CHUNKS")
    print("=" * 60)

    for i, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"\nChunk {i}"
            f"\nPage: {chunk['metadata'].get('page')}"
            f"\nDistance: "
            f"{chunk['distance']:.4f}"
            f"\nText: {chunk['text'][:500]}"
        )

    print("\n" + "=" * 60)
    print("OLLAMA ANSWER")
    print("=" * 60)

    print(answer)