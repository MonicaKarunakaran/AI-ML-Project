from langchain.memory import ConversationBufferMemory


def create_memory():
    return ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )


def add_turn(memory, user_input: str, ai_response: str):
    memory.save_context(
        {"input": user_input},
        {"output": ai_response},
    )


def get_history(memory):
    return memory.load_memory_variables({})["history"]