import pandas as pd

from src.cache import SemanticCache


# ---------------------------------------------------------
# 20 Q&A pairs
# ---------------------------------------------------------

qa_pairs = [
    (
        "What is the purpose of human life?",
        "Human life provides an opportunity to develop knowledge, consciousness and spiritual understanding."
    ),
    (
        "Why should a person perform their duty?",
        "A person should perform their duty responsibly without excessive attachment to the result."
    ),
    (
        "What is karma?",
        "Karma refers to action and its consequences."
    ),
    (
        "Why is attachment to results discouraged?",
        "Attachment to results can create anxiety and emotional dependence on success or failure."
    ),
    (
        "What is spiritual development?",
        "Spiritual development involves knowledge, discipline, consciousness and devotion."
    ),
    (
        "What is the importance of knowledge?",
        "Knowledge helps a person understand their responsibilities and the nature of consciousness."
    ),
    (
        "Why is devotion important?",
        "Devotion helps connect one's activities with a higher spiritual purpose."
    ),
    (
        "Should people avoid all action?",
        "No. People should perform their responsibilities rather than avoiding action."
    ),
    (
        "How can a person reduce attachment?",
        "A person can reduce attachment by focusing on duty rather than being controlled by expected results."
    ),
    (
        "What is selfless action?",
        "Selfless action is performed without excessive concern for personal rewards."
    ),
    (
        "How are knowledge and action connected?",
        "Knowledge provides understanding while disciplined action applies that understanding in practical life."
    ),
    (
        "Why should a person control desire?",
        "Excessive desire can produce attachment, anger, confusion and poor judgment."
    ),
    (
        "What happens when someone depends on results?",
        "Their emotional state can become strongly affected by success or failure."
    ),
    (
        "What does performing duty develop?",
        "Performing duty with the right attitude can develop discipline and mental stability."
    ),
    (
        "What is consciousness?",
        "Consciousness refers to awareness and the mental orientation through which a person experiences life."
    ),
    (
        "What is the role of discipline?",
        "Discipline helps a person perform responsibilities consistently and maintain mental stability."
    ),
    (
        "Can ordinary work support spiritual development?",
        "Yes. Ordinary activities can support spiritual development when performed with appropriate intention."
    ),
    (
        "Why is success not completely under human control?",
        "External circumstances influence outcomes even when a person controls their own effort."
    ),
    (
        "What should a person focus on?",
        "A person should focus on performing their responsibilities sincerely and appropriately."
    ),
    (
        "How are duty and spiritual growth related?",
        "Performing duty without selfish attachment can support discipline, knowledge and spiritual growth."
    ),
]


# ---------------------------------------------------------
# Create cache
# ---------------------------------------------------------

cache = SemanticCache(
    threshold=0.92,
    ttl=3600,
)


# ---------------------------------------------------------
# Store 20 Q&A pairs
# ---------------------------------------------------------

for question, answer in qa_pairs:

    cache.set(
        question,
        answer,
    )


print("\n" + "=" * 70)
print("W5D6 — SEMANTIC CACHE DEMO")
print("=" * 70)

print(
    f"Stored Q&A pairs: {cache.size()}"
)

print(
    "Similarity threshold: 0.92"
)


# ---------------------------------------------------------
# Test queries
# ---------------------------------------------------------

test_queries = [
    "What is the purpose of human existence?",
    "Why should we perform our responsibilities?",
    "Explain karma and its consequences.",
    "Why should people avoid attachment to outcomes?",
    "How does devotion help spiritual progress?",
    "What is the meaning of a completely unrelated question about weather?",
]


results = []

for query in test_queries:

    result = cache.get(query)

    if result:

        print("\nCACHE HIT")
        print(
            f"Query: {query}"
        )
        print(
            f"Matched: {result['matched_query']}"
        )
        print(
            f"Similarity: {result['similarity']:.4f}"
        )

        results.append(
            {
                "Query": query,
                "Cache Hit": True,
                "Similarity": round(
                    result["similarity"],
                    4,
                ),
                "Matched Query": result[
                    "matched_query"
                ],
            }
        )

    else:

        print("\nCACHE MISS")
        print(
            f"Query: {query}"
        )

        results.append(
            {
                "Query": query,
                "Cache Hit": False,
                "Similarity": 0.0,
                "Matched Query": "",
            }
        )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

df = pd.DataFrame(
    results
)

df.to_csv(
    "semantic_cache_results.csv",
    index=False,
)

print("\n" + "=" * 70)

print(
    "Results saved to: semantic_cache_results.csv"
)