"""Part 3 — two separate kinds of memory, both saved on disk.

1. Conversation memory  -> LangGraph checkpointer (SqliteSaver), key = thread_id
   local-storage/conversations.sqlite
   Holds the full graph state of ONE conversation: messages, tool results, tool_rounds.

2. User memory          -> my own small SQLite table, key = user_id
   local-storage/user_memory.sqlite
   Holds saved preferences of ONE person, e.g. {"language": "Kazakh"}.
   It is shared by all conversations of that user and is never shared between users.

Retrieval (find_instructions reading data/helpdesk.json) is NOT memory:
it is fixed reference data, the same for every user and every conversation.
"""
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_KEYS = {"language"}  # only safe, simple preferences; never secrets


def storage_dir() -> Path:
    p = Path(os.getenv("HELPDESK_STORAGE_DIR", "local-storage"))
    return p if p.is_absolute() else ROOT / p


def open_checkpointer(path: Path | None = None) -> SqliteSaver:
    """Conversation memory. The application passes thread_id in the config."""
    path = Path(path) if path else storage_dir() / "conversations.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def reset_storage(path: Path | None = None) -> None:
    """Delete all saved conversations and user preferences."""
    shutil.rmtree(Path(path) if path else storage_dir(), ignore_errors=True)


class UserMemory:
    """User memory: preferences stored per user_id."""

    def __init__(self, path: Path | None = None):
        path = Path(path) if path else storage_dir() / "user_memory.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS preferences (
                               user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                               updated_at TEXT NOT NULL, PRIMARY KEY (user_id, key))""")
        self.conn.commit()

    def get(self, user_id: str) -> dict:
        rows = self.conn.execute("SELECT key, value FROM preferences WHERE user_id = ?",
                                 (user_id,)).fetchall()
        return dict(rows)

    def set(self, user_id: str, key: str, value: str) -> dict:
        key = (key or "").strip().lower()
        value = (value or "").strip()
        if key not in ALLOWED_KEYS:
            return {"saved": False, "error": f"Only these preferences can be saved: {sorted(ALLOWED_KEYS)}."}
        if not value or len(value) > 40:
            return {"saved": False, "error": "The value must be a short word, e.g. 'Kazakh'."}
        self.conn.execute("INSERT OR REPLACE INTO preferences VALUES (?, ?, ?, ?)",
                          (user_id, key, value, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()
        return {"saved": True, "user_id": user_id, "key": key, "value": value,
                "source": "user memory (local-storage/user_memory.sqlite)"}

    def close(self) -> None:
        self.conn.close()
