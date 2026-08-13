from src.chromadb_client import (
    ChromaDBManager,
    create_demo_documents,
)


def main():

    manager = ChromaDBManager()

    # Add the 20 demo documents
    documents, ids, metadatas = (
        create_demo_documents()
    )

    manager.add_documents(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    print("=" * 60)
    print("Semantic Search Demo")
    print("=" * 60)

    print(
        f"Documents available: "
        f"{manager.count()}"
    )

    while True:

        query = input(
            "\nEnter your query "
            "(or 'exit'): "
        )

        if query.lower() == "exit":
            print("Goodbye!")
            break

        results = manager.similarity_search(
            query=query,
            n_results=3,
        )

        print("\nTop 3 results:")

        for i in range(
            len(results["documents"][0])
        ):

            print(
                f"\nResult {i + 1}"
            )

            print(
                "Topic:",
                results["metadatas"][0][i][
                    "topic"
                ],
            )

            print(
                "Distance:",
                round(
                    results["distances"][0][i],
                    4,
                ),
            )

            print(
                "Text:",
                results["documents"][0][i],
            )


if __name__ == "__main__":
    main()