"""
database.py — Simple SQLite helper for chat history.

Tables:
  chats    → stores each conversation (chat_id, title, domain, created_at)
  messages → stores every message      (chat_id, role, content, timestamp)
"""

import sqlite3
import os
from datetime import datetime

# Database file lives next to this script
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")


# ---------- connection helper ----------

def get_connection():
    """Return a new SQLite connection (auto-creates the file if missing)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # so we can access columns by name
    return conn


# ---------- table creation ----------

def init_database():
    """Create the tables if they don't exist yet. Migrate if needed."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL,
            domain     TEXT NOT NULL DEFAULT 'medical',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id    INTEGER NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            timestamp  TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(chat_id)
        )
    """)

    # ── Migration: add domain column if missing (for existing databases) ──
    cursor.execute("PRAGMA table_info(chats)")
    columns = [col[1] for col in cursor.fetchall()]
    if "domain" not in columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN domain TEXT NOT NULL DEFAULT 'medical'")

    conn.commit()
    conn.close()


# ---------- chat helpers ----------

def create_chat(title: str, domain: str = "medical") -> int:
    """Insert a new chat and return its chat_id."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO chats (title, domain, created_at) VALUES (?, ?, ?)",
        (title, domain, now),
    )
    conn.commit()
    chat_id = cursor.lastrowid
    conn.close()
    return chat_id


def get_all_chats() -> list:
    """Return all chats ordered newest-first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chats ORDER BY created_at DESC")
    chats = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return chats


def update_chat_title(chat_id: int, title: str):
    """Update the title of an existing chat."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (title, chat_id))
    conn.commit()
    conn.close()


def update_chat_domain(chat_id: int, domain: str):
    """Update the domain of an existing chat."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE chats SET domain = ? WHERE chat_id = ?", (domain, chat_id))
    conn.commit()
    conn.close()


def delete_chat(chat_id: int):
    """Delete a chat and all its messages."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()


# ---------- message helpers ----------

def save_message(chat_id: int, role: str, content: str):
    """Append a message to a chat."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, now),
    )
    conn.commit()
    conn.close()


def get_messages(chat_id: int) -> list:
    """Return all messages for a chat in chronological order."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
        (chat_id,),
    )
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return messages
