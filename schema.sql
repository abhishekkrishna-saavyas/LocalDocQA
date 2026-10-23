-- ============================================================
-- LocalDocQA — Multi-user schema
-- Adds: users, sessions, conversations, chat_history
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL DEFAULT '',
    username        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    chroma_collection TEXT NOT NULL,      -- e.g. "user_3_docs", set at signup
    default_mode    TEXT NOT NULL DEFAULT 'strict',  -- "strict" or "hybrid"
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    session_token   TEXT PRIMARY KEY,     -- random opaque token, not a JWT
    user_id         INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at      TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- One row per named conversation (ChatGPT-style sidebar entry)
CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    title           TEXT NOT NULL DEFAULT 'New chat',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Individual messages within a conversation
CREATE TABLE IF NOT EXISTS chat_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    user_id         INTEGER NOT NULL,      -- denormalized for fast filtering/safety checks
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    message          TEXT NOT NULL,
    images          TEXT NOT NULL DEFAULT '[]',  -- JSON list of image_path strings matched to this answer (assistant messages only; '[]' for user messages)
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Images extracted from a user's PDFs/PPTX at ingestion time, keyed by
-- which source file and page/slide they came from. This lets a chat
-- answer look up "any images on the page this chunk came from" using
-- the same (filename, page_number) that already lives in chunk metadata
-- (PyPDFLoader's "page" for PDFs, our own "slide" number for PPTX).
CREATE TABLE IF NOT EXISTS document_images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    filename        TEXT NOT NULL,      -- matches the 'source' tagged on chunks
    page_number     INTEGER NOT NULL,   -- matches chunk metadata's page (PDF) or slide (PPTX)
    image_path      TEXT NOT NULL,      -- path on disk to the extracted image file
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_conv ON chat_history(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_document_images_lookup ON document_images(user_id, filename, page_number);
