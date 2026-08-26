from src.memory import add_turn, create_memory, get_history


def test_memory_stores_conversation():
    memory = create_memory()

    add_turn(
        memory,
        "My name is Monica.",
        "Nice to meet you, Monica.",
    )

    history = get_history(memory)

    assert len(history) == 2
    assert "My name is Monica." in history[0].content
    assert "Nice to meet you, Monica." in history[1].content


def test_memory_keeps_multiple_turns():
    memory = create_memory()

    for i in range(5):
        add_turn(
            memory,
            f"User question {i + 1}",
            f"AI response {i + 1}",
        )

    history = get_history(memory)

    assert len(history) == 10