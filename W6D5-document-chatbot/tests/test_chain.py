from langchain_core.prompts import PromptTemplate


def test_prompt_rendering():
    prompt = PromptTemplate(
        template=(
            "History:\n{history}\n"
            "Question:\n{question}"
        ),
        input_variables=["history", "question"],
    )

    result = prompt.format(
        history="User likes Python.",
        question="What language does the user like?",
    )

    assert "Python" in result
    assert "What language" in result


def test_memory_stores_conversation(chain_and_memory):
    _, memory = chain_and_memory

    memory.save_context(
        {"question": "My name is Monica."},
        {"answer": "Nice to meet you, Monica."},
    )

    history = memory.load_memory_variables({})["history"]

    assert "Monica" in history


def test_memory_multiple_turns(chain_and_memory):
    _, memory = chain_and_memory

    conversations = [
        ("My name is Monica.", "Nice to meet you."),
        ("I am learning Python.", "Python is useful."),
        ("I am learning LangChain.", "LangChain is useful."),
        ("I like AI.", "AI is an interesting field."),
        ("I am building a chatbot.", "That sounds like a good project."),
    ]

    for question, answer in conversations:
        memory.save_context(
            {"question": question},
            {"answer": answer},
        )

    history = memory.load_memory_variables({})["history"]

    assert "Monica" in history
    assert "Python" in history
    assert "LangChain" in history
    assert "AI" in history
    assert "chatbot" in history