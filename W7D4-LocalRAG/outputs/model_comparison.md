# Llama3.2:3b vs Qwen2.5:3b Comparison

## Test Setup

Both models were tested locally using Ollama through the LlamaIndex RAG pipeline.

- Model 1: `llama3.2:3b`
- Model 2: `qwen2.5:3b`
- Number of questions: 3
- Same questions were used for both models.
- Same local document context was used.

## Questions

1. What is Retrieval Augmented Generation and why is it useful?
2. Explain the difference between BM25 retrieval and dense retrieval.
3. What is quantisation in LLMs and why is it useful for local inference?

## Observations

### Question 1 — RAG

**llama3.2:3b**

Provided a clear explanation of RAG as a combination of information retrieval and large language models. It explained that relevant documents are retrieved and used as context before generating an answer.

**qwen2.5:3b**

Provided a similar explanation but used more formal wording and emphasized contextual relevance and improving answer quality.

**Observation:** Both models produced relevant answers. Qwen2.5:3b was slightly more detailed.

### Question 2 — BM25 vs Dense Retrieval

**llama3.2:3b**

Provided a clear comparison between traditional BM25 retrieval and vector-based dense retrieval. It explained term frequency, document length, dense vectors, and similarity.

**qwen2.5:3b**

Provided a more technical explanation involving term frequency, inverse document frequency, and dense vector representations.

**Observation:** Both answers were useful. Qwen2.5:3b provided more technical detail, while Llama3.2:3b was easier to read.

### Question 3 — Quantisation

**llama3.2:3b**

Explained that quantisation reduces memory requirements by storing weights with lower precision and makes local inference possible on devices with limited memory.

**qwen2.5:3b**

Provided a more detailed explanation, including examples such as 8-bit and 4-bit precision and how reduced precision lowers memory requirements.

**Observation:** Qwen2.5:3b provided the more complete explanation.

## Overall Comparison

| Criterion | llama3.2:3b | qwen2.5:3b |
|---|---|---|
| Clarity | Very good | Very good |
| Conciseness | Better | Moderate |
| Technical detail | Good | Better |
| Relevance | Good | Good |
| Completeness | Good | Better |
| Overall preference | Good for concise answers | Good for detailed technical answers |

## Conclusion

Both models successfully generated responses for all three test questions.

`llama3.2:3b` generally produced concise and easy-to-understand answers.

`qwen2.5:3b` generally produced more detailed and technically comprehensive answers.

For this test, `qwen2.5:3b` was preferred when more technical detail was required, while `llama3.2:3b` was preferred for concise explanations.