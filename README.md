LocalDocQA
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6446)
A fully local, multi-user Retrieval-Augmented Generation (RAG) app. Sign in, upload your documents, build a vector index, and chat with them. Embeddings and answers are generated locally through Ollama, so there are no cloud calls.
Table of contents
Features
Project structure
Prerequisites
Installation
Usage
Python dependencies
Troubleshooting
Known-good versions
Before you push to GitHub
Features
Fully local: Llama 3.2 for answers and `nomic-embed-text` for embeddings, both served by Ollama
Multi-user accounts with bcrypt-hashed passwords; each user gets their own document folder and their own vector collection
Document support: PDF, DOCX, TXT and PPTX
Two answer modes: Strict (answers only from your documents) and Hybrid (documents first, with clearly labelled general knowledge when they fall short)
Adjustable retrieval: Top-K and temperature sliders
Conversational memory: follow-up questions such as "more details" are rewritten into standalone questions before retrieval
Images from your documents: pictures from the cited PDF page or PPTX slide are shown alongside the answer
ChatGPT-style chat history: named conversations stored in SQLite
Account settings: change name or password, set a default answer mode, manage conversations, or delete the account (which also removes its documents, vectors and chat history)
Project structure
```text
.
├── app.py            # Streamlit web UI (run this)
├── auth.py           # Accounts, sessions, conversations, chat history, image index
├── ingest.py         # Document loading, chunking, embedding, image extraction
├── query.py          # Retrieval and answer generation (Strict / Hybrid)
├── schema.sql        # SQLite schema (must stay next to auth.py)
│
├── localdocqa.db     # created on first run
├── data/             # created on first run
│   └── user_<id>/    #   per-user uploads
│       └── _images/  #   images extracted from PDFs and PPTX files
└── vectorstore/      # created on first run (Chroma, one collection per user)
```
Prerequisites
Python 3.10 or newer. `auth.py` uses the `int | None` type syntax, which does not exist before 3.10. Check with `python --version`.
Ollama, installed and running. On Windows it runs from the system tray after install; otherwise start it with `ollama serve`.
Installation
The commands below are for Windows PowerShell.
1. Clone the repository
```powershell
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```
2. (Optional, recommended) Create a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
> [!TIP]
> If PowerShell says running scripts is disabled, allow it for the current window only and activate again:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
3. Install all Python libraries (one command)
```powershell
python -m pip install --upgrade pip; python -m pip install "streamlit>=1.39" bcrypt langchain-community langchain-core langchain-ollama langchain-text-splitters chromadb pypdf docx2txt python-pptx
```
> [!IMPORTANT]
> Keep the quotes around `"streamlit>=1.39"`. Without them PowerShell treats the `>` as output redirection and the install breaks.
4. Download the Ollama models
```powershell
ollama pull llama3.2; ollama pull nomic-embed-text
```
Model	Role	Approx. size
`llama3.2`	Generates answers (`LLM_MODEL` in `query.py`)	~2 GB
`nomic-embed-text`	Embeddings; required by both `ingest.py` and `query.py`	~0.3 GB
Confirm both are installed with `ollama list`. To use a different LLM, change `LLM_MODEL` near the top of `query.py` (there is a commented-out `qwen2.5` line) and `ollama pull` the matching model first.
Usage
With Ollama running (and the virtual environment active, if you made one):
```powershell
streamlit run app.py
```
The app opens at http://localhost:8501. First-time flow:
Sign up, then log in
Open the Ingestion tab, upload files, and click Run Ingestion
Switch back to the Chat tab, click New chat, and ask questions
Supported uploads are PDF, DOCX, TXT and PPTX. Legacy `.ppt` files are not supported; convert them to `.pptx` first.
Python dependencies
Package	Used by	Purpose
`streamlit` (>=1.39)	`app.py`	Web UI. 1.39+ is required because `app.py` uses `st.container(key=...)`
`bcrypt`	`auth.py`	Password hashing for login and signup
`langchain-community`	`ingest.py`, `query.py`	`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`, and the Chroma wrapper
`langchain-core`	`ingest.py`, `query.py`	`Document`, `ChatPromptTemplate`, `StrOutputParser`
`langchain-text-splitters`	`ingest.py`	`RecursiveCharacterTextSplitter` (chunking)
`langchain-ollama`	`ingest.py`, `query.py`	`OllamaEmbeddings`, `ChatOllama`
`chromadb`	runtime	Vector database. Not imported directly, but the Chroma wrapper fails without it
`pypdf`	`ingest.py`	PDF text loading and PDF image extraction
`docx2txt`	runtime	`.docx` text extraction. Not imported directly; `Docx2txtLoader` needs it
`python-pptx`	`ingest.py`	`.pptx` text and image extraction
The `langchain` meta-package is not needed; no file imports it.
Transitive dependencies (numpy, pydantic, the `ollama` client, Pillow, onnxruntime and so on) are installed automatically by pip.
Standard library modules used (nothing to install): `os`, `sys`, `shutil`, `gc`, `json`, `uuid`, `secrets`, `sqlite3`, `datetime`, `pathlib`, `contextlib`.
Troubleshooting
<details>
<summary><code>No module named ...</code> on startup</summary>
A library is missing. Re-run the install command from step 3, and make sure your virtual environment is activated if you use one.
</details>
<details>
<summary>Errors when embedding or answering, or "connection refused"</summary>
Ollama is not running. Start it (system tray app, or `ollama serve`) and confirm both models appear in `ollama list`.
</details>
<details>
<summary><code>st.container()</code> raises "unexpected keyword argument 'key'"</summary>
Your Streamlit is older than 1.39. Upgrade it:
```powershell
python -m pip install --upgrade "streamlit>=1.39"
```
</details>
<details>
<summary>Database errors mentioning a missing column (<code>name</code>, <code>default_mode</code>, <code>images</code>)</summary>
`localdocqa.db` was created by an older version of the project. The schema only applies to a new database, so close the app, delete `localdocqa.db`, and start again. Accounts and chats in the old file are lost; uploaded files under `data/` are not affected.
</details>
Known-good versions
LangChain packages occasionally rename or move imports between releases. This exact set was installed fresh in a clean virtual environment on Python 3.12, after which the project's modules imported and the login screen rendered without errors:
Package	Version
streamlit	1.66.1
bcrypt	5.0.0
langchain-community	0.4.2
langchain-core	1.7.0
langchain-ollama	1.1.0
langchain-text-splitters	1.1.3
chromadb	1.5.9
pypdf	6.21.2
docx2txt	0.9
python-pptx	1.0.2
<details>
<summary>Reinstall this exact set (if a future update breaks the project)</summary>
```powershell
python -m pip install streamlit==1.66.1 bcrypt==5.0.0 langchain-community==0.4.2 langchain-core==1.7.0 langchain-ollama==1.1.0 langchain-text-splitters==1.1.3 chromadb==1.5.9 pypdf==6.21.2 docx2txt==0.9 python-pptx==1.0.2
```
</details>
Before you push to GitHub
The app creates files that hold user data and credentials (`localdocqa.db` contains password hashes and chat history; `data/` holds uploaded documents). Keep them out of version control by adding a `.gitignore`:
```gitignore
localdocqa.db
data/
vectorstore/
venv/
__pycache__/
```
> [!WARNING]
> If any of these were already committed, adding them to `.gitignore` does not remove them from history. Run `git rm -r --cached <path>` and commit, and consider the contents exposed if the repository was ever public.
