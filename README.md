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


==============================================================================
 QUICK START  (run from the project folder, in this order)
==============================================================================

 1. Install Python 3.10 or newer, and the Ollama app (see section 1).
 2. Install the Python libraries   -> command in section 3.
 3. Download the two Ollama models -> command in section 4.
 4. Start the app:

        streamlit run app.py


==============================================================================
 1. PREREQUISITES (not installed with pip)
==============================================================================

 Python 3.10 or newer
     Required: auth.py uses the "int | None" type syntax, which does not
     exist before Python 3.10. Check with:   python --version

 Ollama (the local model runner)
     Download and install from https://ollama.com/download
     It must be running when you use the app. On Windows it runs in the
     system tray after install; if not, start it with:   ollama serve


==============================================================================
 2. PYTHON LIBRARIES REQUIRED
==============================================================================

 Package               Used by            Purpose
 --------------------  -----------------  ----------------------------------
 streamlit (>=1.39)    app.py             Web UI. Version 1.39+ is REQUIRED:
                                          app.py uses st.container(key=...),
                                          which older versions do not have.
 bcrypt                auth.py            Password hashing for login/signup
 langchain-community   ingest.py, query   PyPDFLoader, Docx2txtLoader,
                                          TextLoader, and the Chroma wrapper
 langchain-core        ingest.py, query   Document, ChatPromptTemplate,
                                          StrOutputParser
 langchain-text-       ingest.py          RecursiveCharacterTextSplitter
   splitters                              (chunking)
 langchain-ollama      ingest.py, query   OllamaEmbeddings, ChatOllama
 chromadb              (runtime)          Vector database. Not imported by
                                          your code directly, but the Chroma
                                          wrapper fails without it.
 pypdf                 ingest.py          PDF text loading and PDF image
                                          extraction (PdfReader)
 docx2txt              (runtime)          .docx text extraction. Not imported
                                          directly; Docx2txtLoader needs it.
 python-pptx           ingest.py          .pptx text and image extraction

 Notes:
   - The "langchain" meta-package is NOT needed. No file imports it.
   - Dependencies of the above (numpy, pydantic, ollama client, Pillow,
     onnxruntime, etc.) are installed automatically by pip.

 Python standard library modules used (nothing to install):
   os, sys, shutil, gc, json, uuid, secrets, sqlite3, datetime, pathlib,
   contextlib


==============================================================================
 3. ONE-COMMAND INSTALL OF ALL PYTHON LIBRARIES  (PowerShell)
==============================================================================

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


==============================================================================
 4. INSTALL THE OLLAMA MODELS
==============================================================================

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


==============================================================================
 5. RUNNING THE APP
==============================================================================

 From the project folder (with the virtual environment active, if you made
 one, and Ollama running):

     streamlit run app.py

 It opens in your browser, normally at http://localhost:8501

 Typical first-time flow:
   Sign up -> log in -> switch to the "Ingestion" tab -> upload files ->
   "Run Ingestion" -> switch back to "Chat" -> "New chat" -> ask questions.

 Supported upload types: PDF, DOCX, TXT, PPTX.
 Legacy .ppt files are not supported - convert them to .pptx first.


==============================================================================
 6. TROUBLESHOOTING
==============================================================================

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
 7. KNOWN-GOOD VERSIONS (optional)
==============================================================================

 LangChain packages occasionally rename or move imports between releases.
 This exact set was installed fresh in a clean virtual environment on
 Python 3.12, and the project's modules imported and the app's login screen
 rendered without errors:

     streamlit 1.66.1          bcrypt 5.0.0
     langchain-community 0.4.2   langchain-core 1.7.0
     langchain-ollama 1.1.0      langchain-text-splitters 1.1.3
     chromadb 1.5.9              pypdf 6.21.2
     docx2txt 0.9                python-pptx 1.0.2

 If a future library update ever breaks the project, reinstall this exact
 set with (single line):

python -m pip install streamlit==1.66.1 bcrypt==5.0.0 langchain-community==0.4.2 langchain-core==1.7.0 langchain-ollama==1.1.0 langchain-text-splitters==1.1.3 chromadb==1.5.9 pypdf==6.21.2 docx2txt==0.9 python-pptx==1.0.2

==============================================================================
