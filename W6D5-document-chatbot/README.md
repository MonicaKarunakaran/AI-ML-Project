# W6D5 - Document Chatbot with LangChain

## Overview

This project implements a local document chatbot workflow using LangChain and Ollama.

The project demonstrates:

- LangChain chains
- PromptTemplate
- Ollama LLM
- Structured output parsing
- ConversationBufferMemory
- LangChain agents
- Calculator tool
- Web-search stub
- PyTest testing
- MLflow integration
- Optional RAGAS evaluation

## Architecture

```text
User
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
Response

Memory
User Input
    |
    v
ConversationBufferMemory
    |
    v
Prompt + History
    |
    v
Ollama
    |
    v
Response

