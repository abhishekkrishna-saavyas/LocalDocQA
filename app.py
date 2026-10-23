"""
app.py
-------
Streamlit UI for the local RAG document Q&A system — multi-user.

UI layout (this revision restructures the frontend only — no backend,
RAG pipeline, LLM, vector store, or state-management logic was changed;
existing functions from auth.py / ingest.py / query.py are wrapped into
the new layout unchanged):

- Sidebar, top to bottom:
    1. Fixed top section: Answer Mode radio + Top-K / Temperature sliders
    2. A horizontal Chat / Ingestion mode toggle
    3. A dynamic middle section that shows either the chat list or the
       document-ingestion controls, depending on the toggle
    4. A profile footer (name, settings gear, logout) pinned to the
       bottom of the sidebar via a CSS position hack (see NOTE below)
- Main frame: welcome header, then either the chat interface or an
  ingestion status panel, depending on the same toggle. The separate
  full-page Settings screen (profile / password / conversations /
  default mode / delete account) still overrides everything else,
  exactly as before.

NOTE on the pinned sidebar footer: Streamlit doesn't officially support
fixed/sticky content inside the sidebar. This uses st.container(key=...)
(Streamlit >= 1.32) to get a stable CSS hook (`div.st-key-profile_footer`)
and pins it to the bottom with `position: sticky` + `margin-top: auto`
inside a flex column — not `position: fixed`, which anchors to the
browser viewport rather than the sidebar and previously caused the
footer to overlap the main chat area. Sticky positioning stays within
the sidebar's own box, so its width always matches the sidebar's actual
rendered width automatically, even if the sidebar is resized. This is
still a DOM/CSS hack, not a supported
API — if a future Streamlit version changes the sidebar's internal
markup, this may need adjusting. If it ever breaks, the safest fallback
is to delete the injected <style> block and let the footer render
in-flow at the bottom of the sidebar's normal content instead.

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit langchain langchain-community langchain-ollama
                langchain-text-splitters chromadb pypdf docx2txt bcrypt
"""

import os
import shutil
import gc
import json
import streamlit as st

from auth import AuthManager
from ingest import (
    VECTORSTORE_DIR,
    EMBED_MODEL,
    get_user_data_dir,
    load_documents,
    split_documents,
    embed_and_store,
    delete_user_files,
    extract_images,
)
from query import (
    load_vectorstore,
    build_qa_engine,
    LLM_MODEL,
    TOP_K,
    DEFAULT_TEMPERATURE,
    IMAGE_MATCH_MAX_DISTANCE,
)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Local RAG Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "localdocqa.db"
auth = AuthManager(DB_PATH)
auth.init_db()


# ======================================================================
# AUTH GATE — nothing below runs until the user is logged in
# (unchanged backend/auth logic — only wrapped as before)
# ======================================================================

def login_signup_screen():
    st.title("🧠 Local RAG Document Q&A")
    st.caption("Sign in to access your documents and chat history.")

    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            user_id = auth.verify_user(username, password)
            if user_id:
                token = auth.create_session(user_id)
                profile = auth.get_user_profile(user_id)
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.session_state.name = profile["name"] if profile else username
                st.session_state.session_token = token
                st.session_state.conversation_id = None
                st.session_state.mode = profile["default_mode"] if profile else "strict"
                st.session_state.view = "chat"
                st.session_state.sidebar_mode = "chat"
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        with st.form("signup_form"):
            new_name = st.text_input("Name", key="signup_name")
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Choose a password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            signup_submitted = st.form_submit_button("Create account", use_container_width=True)

        if signup_submitted:
            if not new_name or not new_username or not new_password:
                st.error("Name, username, and password are required.")
            elif new_password != confirm_password:
                st.error("Passwords don't match.")
            else:
                new_user_id = auth.create_user(new_name, new_username, new_password)
                if new_user_id is None:
                    st.error("That username is already taken.")
                else:
                    st.success("Account created! Switch to the Log in tab to sign in.")


if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    login_signup_screen()
    st.stop()  # nothing past this point executes for a logged-out visitor


# ======================================================================
# LOGGED-IN STATE — everything below is scoped to st.session_state.user_id
# ======================================================================

USER_ID = st.session_state.user_id
DATA_DIR = get_user_data_dir(USER_ID)             # e.g. data/user_3/
COLLECTION_NAME = auth.get_chroma_collection(USER_ID)  # e.g. user_3_docs

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------- SESSION STATE ----------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = os.path.isdir(VECTORSTORE_DIR) and bool(
        os.listdir(VECTORSTORE_DIR)
    )

if "mode" not in st.session_state:
    st.session_state.mode = "strict"

if "top_k" not in st.session_state:
    st.session_state.top_k = TOP_K

if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULT_TEMPERATURE

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "view" not in st.session_state:
    st.session_state.view = "chat"  # "chat" or "settings" (full-page settings override)

if "sidebar_mode" not in st.session_state:
    st.session_state.sidebar_mode = "chat"  # "chat" or "ingestion" (sidebar/main toggle)

if "last_ingestion_result" not in st.session_state:
    st.session_state.last_ingestion_result = None  # {"success": bool, "message": str}


def try_load_chain():
    """Try to load this user's vector store collection. Returns True on success."""
    try:
        st.session_state.vectorstore = load_vectorstore(COLLECTION_NAME)
        st.session_state.vectorstore_ready = True
        return True
    except Exception:
        st.session_state.vectorstore_ready = False
        st.session_state.vectorstore = None
        return False


def clear_vectorstore():
    """
    Clear THIS USER'S collection only.

    Important multi-user change: VECTORSTORE_DIR is a single persist
    directory shared by every user's Chroma collection. We must only ever
    call delete_collection() (scoped to COLLECTION_NAME) and must NEVER
    shutil.rmtree(VECTORSTORE_DIR) anymore — that would wipe every other
    user's vectors too. The old single-user version of this function did
    an rmtree fallback; that fallback is intentionally removed here.
    """
    try:
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore.delete_collection()
        else:
            temp_store = load_vectorstore(COLLECTION_NAME)
            temp_store.delete_collection()
            del temp_store
    except Exception:
        pass  # nothing to clear, or collection was already empty/missing

    st.session_state.vectorstore = None
    st.session_state.vectorstore_ready = False
    gc.collect()  # release any lingering references before the next ingestion


# Attempt to load an existing vector store on first run
if st.session_state.vectorstore_ready and st.session_state.vectorstore is None:
    try_load_chain()


# ---------------------- CONVERSATION HELPERS ----------------------

def start_new_conversation():
    conv_id = auth.create_conversation(USER_ID, title="New chat")
    st.session_state.conversation_id = conv_id
    st.rerun()


def load_conversation_messages():
    """Pull the current conversation's messages from SQLite for rendering."""
    if st.session_state.conversation_id is None:
        return []
    rows = auth.get_messages(st.session_state.conversation_id, USER_ID)
    messages = []
    for r in rows:
        try:
            images = json.loads(r["images"]) if r["images"] else []
        except (json.JSONDecodeError, TypeError):
            images = []
        messages.append({"role": r["role"], "content": r["message"], "images": images})
    return messages


def auto_title_conversation(conv_id: int, first_question: str):
    title = first_question.strip().replace("\n", " ")
    if len(title) > 40:
        title = title[:40].rsplit(" ", 1)[0] + "..."
    auth.rename_conversation(conv_id, USER_ID, title)


# ---------------------- SETTINGS SCREEN (unchanged) ----------------------

def settings_screen():
    st.title("⚙️ Settings")
    st.caption(f"Manage your account, {st.session_state.name}.")

    if st.button("⬅ Back to chat"):
        st.session_state.view = "chat"
        st.rerun()

    tab_profile, tab_password, tab_conversations, tab_mode, tab_danger = st.tabs(
        ["Profile", "Password", "Conversations", "Default Mode", "Delete Account"]
    )

    # ---- Profile: change name ----
    with tab_profile:
        with st.form("update_name_form"):
            new_name = st.text_input("Name", value=st.session_state.name)
            name_submitted = st.form_submit_button("Save name")
        if name_submitted:
            if new_name.strip():
                auth.update_name(USER_ID, new_name.strip())
                st.session_state.name = new_name.strip()
                st.success("Name updated.")
            else:
                st.error("Name cannot be empty.")

    # ---- Password: reset given current password ----
    with tab_password:
        with st.form("reset_password_form"):
            current_password = st.text_input("Current password", type="password")
            new_password = st.text_input("New password", type="password")
            confirm_new_password = st.text_input("Confirm new password", type="password")
            password_submitted = st.form_submit_button("Update password")
        if password_submitted:
            if not current_password or not new_password:
                st.error("All fields are required.")
            elif new_password != confirm_new_password:
                st.error("New passwords don't match.")
            else:
                success = auth.update_password(USER_ID, current_password, new_password)
                if success:
                    st.success("Password updated.")
                else:
                    st.error("Current password is incorrect.")

    # ---- Conversations: manage / delete ----
    with tab_conversations:
        st.markdown("Manage your saved conversations.")
        conversations = auth.list_conversations(USER_ID)
        if not conversations:
            st.caption("No conversations yet.")
        for conv in conversations:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"- {conv['title']}")
            with col2:
                if st.button("Delete", key=f"settings_delete_conv_{conv['id']}"):
                    auth.delete_conversation(conv["id"], USER_ID)
                    if st.session_state.conversation_id == conv["id"]:
                        st.session_state.conversation_id = None
                    st.rerun()

    # ---- Default answer mode ----
    with tab_mode:
        profile = auth.get_user_profile(USER_ID)
        current_default = profile["default_mode"] if profile else "strict"
        mode_choice = st.radio(
            "Default answer mode for new sessions",
            options=["Strict (documents only)", "Hybrid (documents + general knowledge)"],
            index=0 if current_default == "strict" else 1,
        )
        chosen_mode = "strict" if mode_choice.startswith("Strict") else "hybrid"
        if st.button("Save default mode"):
            auth.update_default_mode(USER_ID, chosen_mode)
            st.session_state.mode = chosen_mode
            st.success("Default mode updated.")

    # ---- Delete account ----
    with tab_danger:
        st.warning(
            "Deleting your account permanently removes your uploaded documents, "
            "your vector data, and all your chat history. This cannot be undone."
        )
        confirm_delete = st.checkbox("I understand this action is permanent.")
        confirm_username = st.text_input("Type your username to confirm deletion")

        if st.button("🗑️ Delete my account", type="primary"):
            if not confirm_delete:
                st.error("Please confirm you understand this action is permanent.")
            elif confirm_username != st.session_state.username:
                st.error("Username does not match.")
            else:
                # 1. Clear this user's embeddings (safe API-based clear, avoids
                #    Windows file-lock issues from a live connection)
                clear_vectorstore()
                # 2. Remove this user's raw uploaded files
                delete_user_files(USER_ID)
                # 3. Delete the user row — sessions, conversations, and
                #    chat_history all cascade-delete automatically
                auth.delete_user(USER_ID)
                auth.logout(st.session_state.session_token)

                for key in list(st.session_state.keys()):
                    st.session_state.pop(key, None)

                st.success("Your account has been deleted.")
                st.rerun()


# ---------------------- SIDEBAR ----------------------

# CSS for the sticky profile footer (see module docstring NOTE above) and
# for tighter, cleaner sidebar typography.
st.markdown(
    """
    <style>
    /* Make the sidebar's content column a full-height flex container so
       the footer (margin-top:auto) always lands at the bottom, and use
       position:sticky (not fixed) so its width always matches the
       sidebar's actual rendered width — this is what fixes the overlap
       into the main chat area that position:fixed + a hardcoded width
       caused before. */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        padding-top: 1rem;
        padding-bottom: 0rem !important;
    }

    /* Tighten the gap around the divider between the retrieval sliders
       and the "New chat" button. */
    div.st-key-chat_list_divider hr {
        margin: 0.25rem 0;
    }
    div.st-key-profile_footer {
        margin-top: auto;
        position: sticky;
        bottom: 0;
        border-top: 1px solid var(--border-color, rgba(49, 51, 63, 0.2));
        padding: 0.6rem 0.25rem 0.9rem 0.25rem;
        z-index: 999;
    }

    /* Small uppercase section labels, matching the reference UI */
    .sidebar-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9ca3af;
        font-weight: 600;
        margin: 0.3rem 0 0.35rem 0;
    }

    /* Profile name / "Profile" title — slightly larger than the default
       sidebar text, and rendered as raw HTML (not markdown) so the
       generic .stMarkdown p font-size rule below doesn't shrink it. */
    .profile-name {
        font-size: 1.05rem;
        font-weight: 700;
    }

    /* Slightly tighter, more compact default text sizing in the sidebar */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption {
        font-size: 0.85rem;
    }
    section[data-testid="stSidebar"] .stButton button {
        font-size: 0.85rem;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }

    /* Tighten the gap around the divider between the Chat/Ingestion toggle
       and the section below it (Answer Mode in Chat mode, Document
       Manager in Ingestion mode) — same technique as chat_list_divider
       above: target the <hr>'s own margin rather than the container. */
    div.st-key-tab_toggle_divider hr {
        margin: 0.25rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    # ---- 1. Mode toggle: Chat vs Ingestion (real buttons, not a radio) ----
    with st.container(key="tab_toggle_row"):
        col_tab_chat, col_tab_ingest = st.columns(2)
        with col_tab_chat:
            if st.button(
                "💬 Chat",
                use_container_width=True,
                type="primary" if st.session_state.sidebar_mode == "chat" else "secondary",
                key="tab_chat_btn",
            ):
                st.session_state.sidebar_mode = "chat"
                st.rerun()
        with col_tab_ingest:
            if st.button(
                "📥 Ingestion",
                use_container_width=True,
                type="primary" if st.session_state.sidebar_mode == "ingestion" else "secondary",
                key="tab_ingest_btn",
            ):
                st.session_state.sidebar_mode = "ingestion"
                st.rerun()

    with st.container(key="tab_toggle_divider"):
        st.divider()

    # ---- 2. Dynamic middle section ----
    if st.session_state.sidebar_mode == "chat":
        # -- Answer Mode + Retrieval Parameters (Chat mode only) --
        st.markdown("<div class='sidebar-label'>🎛️ Answer Mode</div>", unsafe_allow_html=True)
        mode_choice = st.radio(
            "How should answers be generated?",
            options=["Strict (documents only)", "Hybrid (documents + general knowledge)"],
            index=0 if st.session_state.mode == "strict" else 1,
            label_visibility="collapsed",
        )
        st.session_state.mode = "strict" if mode_choice.startswith("Strict") else "hybrid"

        st.toggle("Advanced", key="advanced_toggle", value=False)

        if st.session_state.advanced_toggle:
            st.slider(
                "Top-K retrieval",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                step=1,
                key="top_k",
                help="How many document chunks are retrieved to answer each question.",
            )

            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                key="temperature",
                help="LLM generation randomness. Lower = more focused; higher = more varied.",
            )

        with st.container(key="chat_list_divider"):
            st.divider()

        # -- Chat list (ChatGPT-style) --
        if st.button("➕ New chat", use_container_width=True):
            start_new_conversation()

        st.markdown("<div class='sidebar-label'>Recent</div>", unsafe_allow_html=True)
        conversations = auth.list_conversations(USER_ID)
        if conversations:
            for conv in conversations:
                is_active = conv["id"] == st.session_state.conversation_id
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    if st.button(
                        ("📌 " if is_active else "💬 ") + conv["title"],
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        st.session_state.conversation_id = conv["id"]
                        st.rerun()
                with col2:
                    if st.button("✕", key=f"delete_conv_{conv['id']}", help="Delete this conversation"):
                        auth.delete_conversation(conv["id"], USER_ID)
                        if st.session_state.conversation_id == conv["id"]:
                            st.session_state.conversation_id = None
                        st.rerun()
        else:
            st.caption("No conversations yet — click **New chat** to start one.")

    else:
        # -- Ingestion controls --
        st.markdown("<div class='sidebar-label'>Document Manager</div>", unsafe_allow_html=True)
        st.markdown(f"**Embedding model:** `{EMBED_MODEL}`")
        st.markdown(f"**LLM model:** `{LLM_MODEL}`")
        st.markdown(f"**Your collection:** `{COLLECTION_NAME}`")

        st.markdown("")  # small spacer

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "pptx"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} file(s) to your document folder.")

        existing_files = (
            [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
            if os.path.isdir(DATA_DIR)
            else []
        )
        if existing_files:
            st.markdown("**Your files:**")
            for f in existing_files:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.markdown(f"- {f}")
                with col2:
                    if st.button("✕", key=f"delete_file_{f}", help=f"Remove {f}"):
                        # Clean up this file's extracted images (DB rows +
                        # files on disk) so they don't linger as orphans.
                        for img_path in auth.get_images_for_file(USER_ID, f):
                            if os.path.isfile(img_path):
                                os.remove(img_path)
                        auth.delete_images_for_file(USER_ID, f)

                        os.remove(os.path.join(DATA_DIR, f))
                        st.rerun()
        else:
            st.info("No documents uploaded yet.")

        if st.button("⚙️ Run Ingestion", use_container_width=True):
            if not existing_files:
                st.error("No documents to ingest. Upload files first.")
                st.session_state.last_ingestion_result = {
                    "success": False,
                    "message": "No documents to ingest. Upload files first.",
                }
            else:
                with st.spinner("Rebuilding your vector store from current files..."):
                    try:
                        # Clear this user's old collection first so it always
                        # matches exactly what's currently in their data folder.
                        clear_vectorstore()

                        # Also wipe old extracted images (files + DB rows) so
                        # the image table stays in sync with the vector store
                        # on every rebuild, same as the vector store itself.
                        for img_path in auth.get_all_images(USER_ID):
                            if os.path.isfile(img_path):
                                os.remove(img_path)
                        auth.clear_all_images(USER_ID)

                        documents = load_documents(DATA_DIR)
                        chunks = split_documents(documents)
                        embed_and_store(chunks, VECTORSTORE_DIR, EMBED_MODEL, COLLECTION_NAME)

                        # Extract images from the current files and register
                        # each one against the (filename, page) it came from.
                        image_records = extract_images(DATA_DIR, USER_ID)
                        for record in image_records:
                            auth.add_image(
                                USER_ID,
                                record["filename"],
                                record["page_number"],
                                record["image_path"],
                            )

                        st.success(
                            f"Ingestion complete: {len(chunks)} chunks stored, "
                            f"{len(image_records)} image(s) extracted."
                        )
                        st.session_state.last_ingestion_result = {
                            "success": True,
                            "message": f"Ingestion complete: {len(chunks)} chunks stored, "
                            f"{len(image_records)} image(s) extracted from "
                            f"{len(existing_files)} file(s).",
                        }
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
                        st.session_state.last_ingestion_result = {
                            "success": False,
                            "message": f"Ingestion failed: {e}",
                        }

                # Reload the chain with the fresh vector store
                if try_load_chain():
                    st.success("Vector store loaded. Ready to chat!")

        if st.button("🧹 Reset Vector Store", use_container_width=True):
            clear_vectorstore()
            st.success("Your vector store was cleared. Upload files and run ingestion to start fresh.")
            st.session_state.last_ingestion_result = {
                "success": True,
                "message": "Vector store cleared. Upload files and run ingestion to start fresh.",
            }
            st.rerun()

    # Small fixed gap before the profile section (footer's own CSS untouched)
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ---- 4. Sticky bottom profile section ----
    with st.container(key="profile_footer"):
        st.caption("Profile")
        col_user, col_settings = st.columns([4, 1])
        with col_user:
            st.markdown(f"<div class='profile-name'>{st.session_state.name}</div>", unsafe_allow_html=True)
            st.caption(f"@{st.session_state.username}")
        with col_settings:
            if st.button("⚙️", help="Settings", key="open_settings"):
                st.session_state.view = "settings"
                st.rerun()

        if st.button("Log out", use_container_width=True, key="logout_btn"):
            auth.logout(st.session_state.session_token)
            for key in ("user_id", "username", "name", "session_token", "conversation_id",
                        "vectorstore", "vectorstore_ready", "mode", "view", "sidebar_mode",
                        "top_k", "temperature", "last_ingestion_result"):
                st.session_state.pop(key, None)
            st.rerun()


# ---------------------- MAIN CONTENT ----------------------

# Full-page Settings overrides everything else, same as before.
if st.session_state.view == "settings":
    settings_screen()
    st.stop()

st.title(f"👋 Welcome, {st.session_state.name}")
st.caption("Local RAG Document Q&A")

if st.session_state.sidebar_mode == "ingestion":
    # ---- Ingestion status view ----
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vector store", "Ready ✅" if st.session_state.vectorstore_ready else "Not built ⚠️")
    with col2:
        file_count = (
            len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
            if os.path.isdir(DATA_DIR)
            else 0
        )
        st.metric("Files in your folder", file_count)

    last_result = st.session_state.last_ingestion_result
    if last_result:
        if last_result["success"]:
            st.success(last_result["message"])
        else:
            st.error(last_result["message"])
    else:
        st.info(
            "Upload documents and click **⚙️ Run Ingestion** in the sidebar "
            "to build your vector store."
        )

else:
    # ---- Chat view ----
    if st.session_state.conversation_id is None:
        st.info("Select a conversation from the sidebar, or click **New chat** to start one.")
    elif not st.session_state.vectorstore_ready:
        st.warning(
            "No vector store found yet. Switch to **📥 Ingestion** in the sidebar "
            "to upload documents and get started."
        )
    else:
        messages = load_conversation_messages()

        # Render chat history for the active conversation
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                for img_path in msg.get("images", []):
                    if os.path.isfile(img_path):
                        st.image(img_path, width=300)

        # Chat input
        question = st.chat_input("Ask a question about your documents...")

        if question:
            current_mode = st.session_state.mode
            conv_id = st.session_state.conversation_id
            is_first_message = len(messages) == 0

            # History for contextualizing follow-ups = prior turns in THIS conversation only
            prior_history = messages

            auth.add_message(conv_id, USER_ID, "user", question)
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        qa = build_qa_engine(
                            st.session_state.vectorstore,
                            mode=current_mode,
                            top_k=st.session_state.top_k,
                            temperature=st.session_state.temperature,
                        )
                        answer, docs = qa(question, prior_history)
                        sources = sorted(
                            set(doc.metadata.get("source", "unknown") for doc in docs)
                        )

                        # Find any images extracted from the same
                        # (filename, page/slide) as the retrieved chunks, so
                        # they can be shown alongside this answer.
                        matched_images = []
                        seen_image_paths = set()
                        for doc in docs:
                            # Retrieval always returns the top-K nearest chunks
                            # even when none are truly relevant, so only chunks
                            # close enough to the question may pull in images.
                            score = doc.metadata.get("score")
                            if score is None or score > IMAGE_MATCH_MAX_DISTANCE:
                                continue
                            filename = doc.metadata.get("source")
                            page_number = doc.metadata.get("page")
                            if page_number is None:
                                page_number = doc.metadata.get("slide")
                            if filename is None or page_number is None:
                                continue
                            for img_path in auth.get_images_for_page(USER_ID, filename, page_number):
                                if img_path not in seen_image_paths and os.path.isfile(img_path):
                                    seen_image_paths.add(img_path)
                                    matched_images.append(img_path)
                    except Exception as e:
                        answer = f"Error while generating answer: {e}"
                        sources = []
                        matched_images = []

                st.markdown(answer)
                for img_path in matched_images:
                    st.image(img_path, width=300)
                if sources:
                    mode_label = "🔒 Strict" if current_mode == "strict" else "🔀 Hybrid"
                    with st.expander(f"📚 Sources used · {mode_label}"):
                        for src in sources:
                            st.markdown(f"- {src}")

            auth.add_message(conv_id, USER_ID, "assistant", answer, images=matched_images)

            if is_first_message:
                auto_title_conversation(conv_id, question)

            st.rerun()
