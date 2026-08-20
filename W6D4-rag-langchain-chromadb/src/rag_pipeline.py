"""
Exercise 3:
End-to-end RAG pipeline using LangChain, ChromaDB and Ollama.

Pipeline:
PDF -> chunks -> embeddings -> ChromaDB -> top-3 retrieval
-> prompt -> Ollama -> final answer
"""

from pathlib import Path
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "sample.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "w6d4_rag_documents"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"


def load_pdf():
    """Load text from the PDF."""

    print("\nLoading PDF...")
    print(f"File: {PDF_PATH}")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Please place your PDF inside the data/ folder "
            "and rename it to sample.pdf."
        )

    reader = PdfReader(str(PDF_PATH))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    print(f"Pages with text: {len(pages)}")

    return pages


def split_documents(pages):
    """Split PDF text into smaller chunks."""

    print("\nSplitting PDF into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append(
                {
                    "text": chunk,
                    "page": page["page"],
                }
            )

    print(f"Created {len(chunks)} chunks.")

    return chunks


def store_documents(chunks, embeddings):
    """Create ChromaDB collection and store embeddings."""

    print("\nCreating ChromaDB collection...")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Existing RAG collection removed.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings in batches...")

    vectors = []

    batch_size = 50

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))

        batch = texts[start:end]

        print(
            f"Embedding chunks {start + 1}-{end} "
            f"of {len(texts)}..."
        )

        batch_vectors = embeddings.embed_documents(batch)
        vectors.extend(batch_vectors)

    ids = [f"pdf_chunk_{i + 1}" for i in range(len(chunks))]

    metadatas = [
        {
            "source": PDF_PATH.name,
            "page": chunk["page"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")

    return client, collection


def retrieve_documents(collection, embeddings, query):
    """Retrieve the top 3 most relevant chunks."""

    print("\nRetrieving top 3 relevant chunks...")
    print(f"Query: {query}")

    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    return results


def build_context(results):
    """Build context from retrieved chunks."""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):
        context_parts.append(
            f"[Chunk {i} | Page {metadata['page']}]\n{document}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query, context):
    """Generate an answer using Ollama through LangChain."""

    print("\nGenerating answer with Ollama...")

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    return response.content


def main():
    print("=" * 70)
    print("W6D4 - LANGCHAIN + CHROMADB + OLLAMA RAG PIPELINE")
    print("=" * 70)

    pages = load_pdf()

    chunks = split_documents(pages)

    if not chunks:
        raise ValueError(
            "No text chunks were extracted from the PDF. "
            "Make sure the PDF contains selectable text."
        )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    _, collection = store_documents(
        chunks,
        embeddings,
    )

    query = input(
        "\nEnter your question about the PDF: "
    ).strip()

    if not query:
        query = "What is the main topic of this document?"

    results = retrieve_documents(
        collection,
        embeddings,
        query,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n" + "-" * 70)
    print("TOP 3 RETRIEVED CHUNKS")
    print("-" * 70)

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print(f"\n### Chunk {i}")
        print(f"Page: {metadata['page']}")
        print(f"Cosine distance: {distance:.4f}")
        print(document[:500])

    context = build_context(results)

    answer = generate_answer(
        query,
        context,
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(answer)

    print("\nRAG pipeline completed successfully.")


if __name__ == "__main__":
    main()