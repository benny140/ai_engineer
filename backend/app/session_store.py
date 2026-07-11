import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4


class SessionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize(self, reset: bool = True) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.db_path.exists():
            self.db_path.unlink()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL DEFAULT '',
                    last_summarized_message_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_id_id ON messages(session_id, id)"
            )
            conn.commit()

    def ensure_session(self, session_id: str | None = None) -> str:
        sid = (session_id or "").strip() or str(uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions(session_id)
                VALUES (?)
                """,
                (sid,),
            )
            conn.commit()
        return sid

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages(session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, role, content),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def get_message_count(self, session_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            )
            row = cur.fetchone()
        return int(row[0] if row else 0)

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id, role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()

        return [
            {"id": int(r[0]), "role": str(r[1]), "content": str(r[2])} for r in rows
        ]

    def get_recent_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        messages = self.get_messages(session_id)
        if limit <= 0:
            return []
        return messages[-limit:]

    def get_summary(self, session_id: str) -> tuple[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT summary_text, last_summarized_message_id
                FROM summaries
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cur.fetchone()

        if not row:
            return "", 0

        return str(row[0] or ""), int(row[1] or 0)

    def upsert_summary(
        self,
        session_id: str,
        summary_text: str,
        last_summarized_message_id: int,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO summaries(session_id, summary_text, last_summarized_message_id, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    last_summarized_message_id = excluded.last_summarized_message_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, summary_text, last_summarized_message_id),
            )
            conn.commit()

    def get_unsummarized_older_messages(
        self,
        session_id: str,
        keep_recent: int,
        last_summarized_message_id: int,
    ) -> list[dict[str, Any]]:
        messages = self.get_messages(session_id)
        if len(messages) <= keep_recent:
            return []

        older = messages[:-keep_recent]
        return [m for m in older if int(m["id"]) > last_summarized_message_id]
