# Changelog

## [0.1.0] - 2026-08-28

### Added

- Set up Ollama for local LLM inference.
- Added support for `llama3.2:3b` and `qwen2.5:3b`.
- Added local `nomic-embed-text` embeddings.
- Implemented LlamaIndex-based local RAG pipeline.
- Added custom AI/ML system prompt.
- Added five-question RAG testing.
- Added llama3.2:3b vs qwen2.5:3b comparison.
- Added model comparison documentation.
- Added PyTest tests.
- Added README documentation.

### Improvements

- Configured the RAG pipeline to build the document index once and
  reuse it across multiple questions.
- Increased local Ollama request timeout to support cold-start inference.

### Testing

- PyTest result: **3 passed**
- Local RAG inference completed successfully for five questions.
- Both local LLMs successfully answered the three comparison questions.