# W9D2 - CrewAI Tools: Web Search and Code Execution

## Objective

Build a multi-agent research crew using CrewAI with web search and
code execution tools.

## Technology Stack

- CrewAI
- Ollama
- Llama 3.2 3B
- Python
- DDGS Web Search
- PyTest

## Agents

### 1. Researcher

Role:
AI Researcher

Responsibilities:

- Search the web
- Collect recent information
- Identify important facts
- Use code execution for simple calculations

### 2. Writer

Role:
Technical Writer

Responsibilities:

- Convert research findings into a report
- Organize information using headings
- Explain technical concepts clearly

### 3. Reviewer

Role:
Research Reviewer

Responsibilities:

- Check accuracy
- Check completeness
- Improve clarity
- Review the final report

## Workflow

```text
Web Search
    |
    v
Researcher
    |
    v
Writer
    |
    v
Reviewer
    |
    v
Final Research Report