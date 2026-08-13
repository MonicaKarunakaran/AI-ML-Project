import mlflow

def start_run(experiment_name: str = "W5D3-ChromaDB-RAG"):

    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()

def log_rag_parameters(
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    embedding_model: str,
    llm_model: str
):
    mlflow.log_params(
        {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
        }
    )


def log_retrieval_metrics(
    number_of_chunks: int,
    retrieval_count: int
):
    mlflow.log_metrics(
        {
            "number_of_chunks": number_of_chunks,
            "retrieval_count": retrieval_count,
        }
    )