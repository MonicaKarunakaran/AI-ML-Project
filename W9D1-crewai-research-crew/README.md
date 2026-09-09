# W9D1 - CrewAI Multi-Agent Research Crew

## Objective

Build a multi-agent research system using CrewAI.

The system contains three agents:

1. Researcher
2. Writer
3. Reviewer

## Architecture

Researcher
    |
    v
Research Task
    |
    v
Writer
    |
    v
Writing Task
    |
    v
Reviewer
    |
    v
Final Improved Report

## Technologies

- Python
- CrewAI
- CrewAI Tools
- Ollama
- Llama 3.2
- Pytest
- Serper Web Search

## Agents

### Researcher

Collects useful and accurate information about the given topic.

### Writer

Converts the research into a clear and structured article.

### Reviewer

Checks the article for accuracy, completeness, clarity and grammar.

## Process

The project uses a sequential CrewAI process.

The order is:

Researcher -> Writer -> Reviewer

## Running the Project

Activate the virtual environment:

```bash
source .venv/Scripts/activate