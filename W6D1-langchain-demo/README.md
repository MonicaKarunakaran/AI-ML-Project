# W6D1 - LangChain Fundamentals

## Overview

This project demonstrates LangChain fundamentals using a locally hosted
Ollama LLM.

The project covers:

- LangChain chains
- PromptTemplate
- Ollama LLM
- OutputParser
- ConversationBufferMemory
- Tool calling
- LangChain AgentExecutor
- Calculator tool
- Web-search stub
- MLflow experiment tracking
- Pytest testing

## Architecture

```text
User Input
    |
    v
PromptTemplate
    |
    v
Ollama LLM
    |
    v
OutputParser
    |
    v
JSON Response