import yaml
import ollama

from src.embedding import get_embeddings
from src.chroma_store import ChromaStore
from src.pdf_utils import extract_text, chunk_text

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def index_pdf(pdf_path: str):

    config = load_config()
    embedding_model = config["ollama"]["embedding_model"]
    persist_directory = config["chroma"]["persist_directory"]
    collection_name = config["chroma"]["collection_name"]
    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]

    text = extract_text(pdf_path)

    if not text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print(f"Extracted characters: {len(text)}")
    print(f"Created chunks: {len(chunks)}")

    embeddings = get_embeddings(
        chunks,
        model=embedding_model
    )

    store = ChromaStore(
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    ids = [
        f"pdf_chunk_{i+1}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": pdf_path,
            "chunk": i + 1
        }
        for i in range(len(chunks))
    ]

    store.add_documents(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return store


def retrieve_context(
    store: ChromaStore,
    question: str,
    top_k: int = 3,
    embedding_model: str = "nomic-embed-text"
):

    query_embedding = get_embeddings(
        [question],
        model=embedding_model
    )[0]

    results = store.query(
        query_embedding=query_embedding,
        n_results=top_k
    )

    documents = results["documents"][0]

    return documents


def generate_answer(
    question: str,
    context: list[str],
    llm_model: str = "llama3.2:3b"
):

    context_text = "\n\n".join(
        f"Context {i + 1}:\n{chunk}"
        for i, chunk in enumerate(context)
    )

    prompt = f"""
You are a helpful question-answering assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Context:
{context_text}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=llm_model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def run_rag(pdf_path: str, question: str):

    config = load_config()

    store = index_pdf(pdf_path)

    context = retrieve_context(
        store=store,
        question=question,
        top_k=config["retrieval"]["top_k"],
        embedding_model=config["ollama"]["embedding_model"]
    )

    answer = generate_answer(
        question=question,
        context=context,
        llm_model=config["ollama"]["llm_model"]
    )

    return {
        "question": question,
        "retrieved_chunks": context,
        "answer": answer
    }