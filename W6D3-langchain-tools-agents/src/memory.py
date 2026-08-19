from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama


def create_memory_chain():
    """Create a conversation chain using an explicit message history."""

    llm = ChatOllama(model="llama3.2:3b")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI/ML learning assistant. "
                "Use the conversation history to answer questions accurately. "
                "When the user asks what they are learning, use previous messages "
                "to identify the topic they explicitly mentioned. "
                "LangChain is a framework for developing applications powered by "
                "language models, including chains, agents, tools, and retrieval.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    return chain


def run_conversation():
    """Run five conversation turns while maintaining message history."""

    chain = create_memory_chain()
    history = []
    results = []

    turns = [
        "My name is Monica.",
        "I am learning LangChain.",
        "What am I currently learning?",
        "What is my name?",
        "Summarize what you know about me from this conversation.",
    ]

    for message in turns:
        response = chain.invoke(
            {
                "input": message,
                "history": history,
            }
        )

        response_text = response.content

        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=response_text))

        results.append((message, response_text))

    return results, history