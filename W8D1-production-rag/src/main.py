from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Production RAG API",
    description="Containerized RAG API for W8D1 MLOps practice",
    version="1.0.0",
)


# Small knowledge base used for the local demonstration.
# In a production system this would normally come from a vector database.
KNOWLEDGE_BASE = [
    {
        "title": "RAG",
        "content": (
            "Retrieval-Augmented Generation combines document retrieval "
            "with a language model. Relevant documents are retrieved first "
            "and then supplied to the language model to generate an answer."
        ),
    },
    {
        "title": "Docker",
        "content": (
            "Docker packages an application and its dependencies into a "
            "container so that it can run consistently across environments."
        ),
    },
    {
        "title": "MLOps",
        "content": (
            "MLOps applies software engineering and DevOps practices to "
            "machine learning systems. It includes testing, deployment, "
            "monitoring, versioning and continuous improvement."
        ),
    },
    {
        "title": "Monitoring",
        "content": (
            "Production ML systems should monitor latency, error rate, "
            "resource utilization, retrieval quality, model quality and "
            "user feedback."
        ),
    },
]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]


def retrieve_documents(question: str, top_k: int = 3) -> List[dict]:
    """Retrieve simple keyword-matching documents.

    This lightweight retriever keeps the project runnable in Docker and CI.
    A production implementation can replace this function with ChromaDB,
    FAISS, Elasticsearch or another vector database.
    """

    question_words = {
        word.lower().strip(".,?!")
        for word in question.split()
        if len(word) > 2
    }

    scored_documents = []

    for document in KNOWLEDGE_BASE:
        text = f"{document['title']} {document['content']}".lower()

        score = sum(1 for word in question_words if word in text)

        scored_documents.append((score, document))

    scored_documents.sort(key=lambda item: item[0], reverse=True)

    return [
        document
        for score, document in scored_documents[:top_k]
        if score > 0
    ]


def generate_answer(question: str, documents: List[dict]) -> str:
    """Generate a deterministic answer for the demo RAG pipeline."""

    if not documents:
        return (
            "I could not find relevant information in the knowledge base "
            "for this question."
        )

    context = " ".join(document["content"] for document in documents)

    return (
        f"Based on the retrieved knowledge: {context} "
        f"Your question was: {question}"
    )


@app.get("/")
def root():
    return {
        "message": "Production RAG API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "production-rag-api",
        "version": "1.0.0",
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    documents = retrieve_documents(request.question)

    answer = generate_answer(
        request.question,
        documents,
    )

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[document["title"] for document in documents],
    )
