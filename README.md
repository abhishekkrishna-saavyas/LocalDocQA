# LocalDocQA

 LocalDocQA - Local RAG Document Q&A
 README: requirements and setup (Windows / PowerShell)


WHAT THIS PROJECT IS
--------------------
A fully local, multi-user Retrieval-Augmented Generation (RAG) app. Users
sign in, upload PDF / DOCX / TXT / PPTX files, build a vector index, and chat
with their documents. Embeddings and answers are generated locally through
Ollama - no cloud calls.

PROJECT FILES
-------------
  app.py       Streamlit web UI (this is the file you run)
  auth.py      Accounts, sessions, conversations, chat history, image index
  ingest.py    Document loading, chunking, embedding, image extraction
  query.py     Retrieval + answer generation (Strict / Hybrid modes)
  schema.sql   SQLite schema. MUST stay in the same folder as auth.py

Created automatically on first run (do not create these by hand):
  localdocqa.db   SQLite database (users, chats, image index)
  data/           Per-user uploads: data/user_<id>/ and data/user_<id>/_images/
  vectorstore/    Chroma vector database (one collection per user)




#PREREQUISITES (not installed with pip)

 Python 3.10 or newer
     Required: auth.py uses the "int | None" type syntax, which does not
     exist before Python 3.10. Check with:   python --version

 Ollama (the local model runner)
     Download and install from https://ollama.com/download
     It must be running when you use the app. On Windows it runs in the
     system tray after install; if not, start it with:   ollama serve


#ONE-COMMAND INSTALL OF ALL PYTHON LIBRARIES  (PowerShell)


 Paste as a single line:

python -m pip install --upgrade pip; python -m pip install "streamlit>=1.39" bcrypt langchain-community langchain-core langchain-ollama langchain-text-splitters chromadb pypdf docx2txt python-pptx

 Keep the quotes around "streamlit>=1.39" - without them PowerShell treats
 the ">" as output redirection and the install breaks.

 OPTIONAL - use a virtual environment (recommended; keeps this project's
 libraries separate from everything else on your machine). Run these first,
 then the install command above:

     python -m venv venv
     .\venv\Scripts\Activate.ps1

 If PowerShell refuses to run the activate script ("running scripts is
 disabled"), allow it for this window only, then activate again:

     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


#INSTALL THE OLLAMA MODELS


 The LLM used for answers (the model set in query.py as LLM_MODEL):

     ollama pull llama3.2

 The embedding model (REQUIRED as well - ingest.py and query.py both use it
 to turn text into vectors; ingestion and search fail without it):

     ollama pull nomic-embed-text

 Both in one line (PowerShell):

     ollama pull llama3.2; ollama pull nomic-embed-text

 Approximate download sizes: llama3.2 about 2 GB, nomic-embed-text about
 0.3 GB. Confirm both are installed with:

     ollama list

 To use a different LLM, change LLM_MODEL near the top of query.py (there is
 a commented-out "qwen2.5" line there) and pull the matching model first,
 e.g.   ollama pull qwen2.5


RUNNING THE APP


 From the project folder (with the virtual environment active, if you made
 one, and Ollama running):

     streamlit run app.py


TROUBLESHOOTING


 "No module named ..." on startup
     A library is missing - re-run the install command in section 3, and
     make sure the virtual environment (if you use one) is activated.

 App errors when embedding or answering / "connection refused"
     Ollama is not running. Start it (system tray app, or: ollama serve)
     and confirm both models appear in:   ollama list

 st.container() "unexpected keyword argument 'key'"
     Your Streamlit is older than 1.39. Fix:
         python -m pip install --upgrade "streamlit>=1.39"

 Database errors mentioning a missing column (name, default_mode, images)
     localdocqa.db was created by an older version of this project. The
     schema only applies to a NEW database, so close the app, delete
     localdocqa.db, and start again. Accounts and chats in the old file are
     lost; uploaded files under data/ are not affected.



==============================================================================
