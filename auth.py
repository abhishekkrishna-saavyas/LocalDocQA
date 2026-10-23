"""
auth.py — Authentication, session, and conversation management for LocalDocQA.

Handles:
  - User signup / login (bcrypt password hashing)
  - Session tokens (opaque, stored in SQLite, checked on each request)
  - Per-user ChromaDB collection naming
  - ChatGPT-style named conversations (create, list, rename, delete)

Usage from app.py:
    from auth import AuthManager

    auth = AuthManager("localdocqa.db")
    auth.init_db()  # run once, safe to call every startup (CREATE TABLE IF NOT EXISTS)

    user_id = auth.verify_user(username, password)
    if user_id:
        token = auth.create_session(user_id)
"""

import sqlite3
import bcrypt
import secrets
import json
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

SESSION_LIFETIME_HOURS = 12
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class AuthManager:
    def __init__(self, db_path: str = "localdocqa.db"):
        self.db_path = db_path

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self):
        """Run once at startup. Safe to call repeatedly."""
        with self._conn() as conn:
            conn.executescript(SCHEMA_PATH.read_text())

    # ---------------- Users ----------------

    def create_user(self, name: str, username: str, password: str) -> int | None:
        """Returns new user_id, or None if username already exists."""
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        with self._conn() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO users (name, username, password_hash, chroma_collection) "
                    "VALUES (?, ?, ?, ?)",
                    (name, username, password_hash, "__placeholder__"),
                )
                user_id = cur.lastrowid
                # Collection name depends on the id, so patch it in after insert
                collection_name = f"user_{user_id}_docs"
                conn.execute(
                    "UPDATE users SET chroma_collection = ? WHERE id = ?",
                    (collection_name, user_id),
                )
                return user_id
            except sqlite3.IntegrityError:
                return None  # username taken

    def verify_user(self, username: str, password: str) -> int | None:
        """Returns user_id if credentials are valid, else None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, password_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
        if row and bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
            return row["id"]
        return None

    def get_chroma_collection(self, user_id: int) -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT chroma_collection FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return row["chroma_collection"] if row else f"user_{user_id}_docs"

    def get_user_profile(self, user_id: int) -> dict | None:
        """Returns {'id', 'name', 'username', 'default_mode'} for the settings page."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, name, username, default_mode FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_name(self, user_id: int, new_name: str):
        with self._conn() as conn:
            conn.execute("UPDATE users SET name = ? WHERE id = ?", (new_name, user_id))

    def update_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """Verifies current_password before setting new_password. Returns False if
        the current password is wrong (and makes no change in that case)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if not row or not bcrypt.checkpw(
                current_password.encode(), row["password_hash"].encode()
            ):
                return False
            new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id)
            )
            return True

    def update_default_mode(self, user_id: int, mode: str):
        if mode not in ("strict", "hybrid"):
            raise ValueError("mode must be 'strict' or 'hybrid'")
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET default_mode = ? WHERE id = ?", (mode, user_id)
            )

    def delete_user(self, user_id: int):
        """Deletes the user row. Sessions, conversations, chat_history, and
        document_images all cascade-delete automatically via their ON DELETE
        CASCADE foreign keys (PRAGMA foreign_keys = ON is set on every
        connection in _conn()). This does NOT touch the user's Chroma
        collection or on-disk files (including extracted image files) —
        the caller (app.py) is responsible for clearing those separately,
        since only it holds the live vector store connection. Extracted
        images specifically live under that user's data folder, though, so
        ingest.py's delete_user_files() already removes them for free."""
        with self._conn() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    # ---------------- Document images ----------------
    # Maps a user's extracted PDF/PPTX images to the (filename, page_number)
    # they came from, so a chat answer can look up "any images on the page
    # this chunk came from" using the same numbering already present in
    # chunk metadata (PyPDFLoader's "page" for PDFs, our own "slide" for
    # PPTX — see ingest.py). Row-only operations: deleting the actual image
    # files on disk is the caller's (app.py's) job, since these methods have
    # no filesystem access by design (matches this module's existing
    # DB-only scope).

    def add_image(self, user_id: int, filename: str, page_number: int, image_path: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO document_images (user_id, filename, page_number, image_path) "
                "VALUES (?, ?, ?, ?)",
                (user_id, filename, page_number, image_path),
            )

    def get_images_for_page(self, user_id: int, filename: str, page_number: int) -> list[str]:
        """Image paths extracted from this exact (filename, page_number) —
        used at chat time to show images matching a retrieved chunk."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT image_path FROM document_images "
                "WHERE user_id = ? AND filename = ? AND page_number = ?",
                (user_id, filename, page_number),
            ).fetchall()
        return [r["image_path"] for r in rows]

    def get_images_for_file(self, user_id: int, filename: str) -> list[str]:
        """Every image path extracted from this filename, any page — used
        when a single file is removed, so its images can be deleted from
        disk before delete_images_for_file() removes the DB rows."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT image_path FROM document_images WHERE user_id = ? AND filename = ?",
                (user_id, filename),
            ).fetchall()
        return [r["image_path"] for r in rows]

    def get_all_images(self, user_id: int) -> list[str]:
        """Every image path for this user — used when Run Ingestion wipes
        and rebuilds everything, so old image files can be deleted from disk
        before clear_all_images() removes the DB rows."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT image_path FROM document_images WHERE user_id = ?", (user_id,)
            ).fetchall()
        return [r["image_path"] for r in rows]

    def delete_images_for_file(self, user_id: int, filename: str):
        """Deletes the DB rows for one file's images. Call
        get_images_for_file() first if the files on disk also need removing."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM document_images WHERE user_id = ? AND filename = ?",
                (user_id, filename),
            )

    def clear_all_images(self, user_id: int):
        """Deletes ALL image rows for this user. Call get_all_images() first
        if the files on disk also need removing."""
        with self._conn() as conn:
            conn.execute("DELETE FROM document_images WHERE user_id = ?", (user_id,))

    # ---------------- Sessions ----------------

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.utcnow() + timedelta(hours=SESSION_LIFETIME_HOURS)).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (session_token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at),
            )
        return token

    def validate_session(self, token: str) -> int | None:
        """Returns user_id if token is valid and unexpired, else None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_id, expires_at FROM sessions WHERE session_token = ?",
                (token,),
            ).fetchone()
            if not row:
                return None
            if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
                conn.execute("DELETE FROM sessions WHERE session_token = ?", (token,))
                return None
            return row["user_id"]

    def logout(self, token: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE session_token = ?", (token,))

    # ---------------- Conversations (ChatGPT-style) ----------------

    def create_conversation(self, user_id: int, title: str = "New chat") -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
                (user_id, title),
            )
            return cur.lastrowid

    def list_conversations(self, user_id: int) -> list[sqlite3.Row]:
        """Most recently updated first — mirrors a typical chat sidebar."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()

    def rename_conversation(self, conversation_id: int, user_id: int, new_title: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
                (new_title, conversation_id, user_id),
            )

    def delete_conversation(self, conversation_id: int, user_id: int):
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )

    # ---------------- Chat history ----------------

    def add_message(
        self,
        conversation_id: int,
        user_id: int,
        role: str,
        message: str,
        images: list | None = None,
    ):
        """
        images: list of image_path strings matched to this answer (assistant
        messages only — leave as None for user messages). Stored as a JSON
        list so get_messages() can hand back the exact images that were
        shown alongside this answer, letting them reappear correctly when
        the conversation is reloaded (not just on the turn they were sent).
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_history (conversation_id, user_id, role, message, images) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, user_id, role, message, json.dumps(images or [])),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )

    def get_messages(self, conversation_id: int, user_id: int) -> list[sqlite3.Row]:
        """Always filter by user_id too, even though conversation_id is unique —
        this is the belt-and-braces check that stops one user reading another's
        conversation via a guessed/forged conversation_id.

        Note: the "images" field in each row is a raw JSON string (as stored
        by add_message) — the caller is responsible for json.loads()-ing it,
        consistent with how this module leaves other reshaping to callers
        (e.g. app.py's load_conversation_messages())."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT role, message, images, created_at FROM chat_history "
                "WHERE conversation_id = ? AND user_id = ? ORDER BY created_at ASC",
                (conversation_id, user_id),
            ).fetchall()
