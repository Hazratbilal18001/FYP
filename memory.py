"""
memory.py — Bridges the SQLite database with LangChain message objects.

Converts stored messages into the format LangGraph agents expect.
"""

from langchain_core.messages import HumanMessage, AIMessage
from database import get_messages, save_message


def load_chat_history(chat_id: int) -> list:
    """
    Load all messages for a chat and convert them to LangChain message objects.

    Returns a list like:
        [HumanMessage("hi"), AIMessage("Hello! How can I help?"), ...]
    """
    rows = get_messages(chat_id)
    history = []

    for row in rows:
        if row["role"] == "user":
            history.append(HumanMessage(content=row["content"]))
        else:
            history.append(AIMessage(content=row["content"]))

    return history


def save_user_message(chat_id: int, content: str):
    """Save a user message to the database."""
    save_message(chat_id, "user", content)


def save_ai_message(chat_id: int, content: str):
    """Save an AI message to the database."""
    save_message(chat_id, "assistant", content)
