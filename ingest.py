"""
ingest.py
----------
Ingestion pipeline for the local RAG project.

What it does:
1. Loads documents from a per-user data folder (PDF, DOCX, TXT, PPTX)
2. Splits them into overlapping chunks
3. Generates embeddings using nomic-embed-text (via Ollama)
4. Stores everything in a persisted Chroma vector store, in a collection
   scoped to that user (collection_name=f"user_{user_id}_docs")
5. Extracts embedded images from PDF/PPTX files (see extract_images())
   so they can be shown alongside chat answers that cite the page/slide
   they appear on. This function only extracts files to disk and returns
   metadata records — it does NOT write to the database. The caller
   (app.py) is responsible for saving those records via auth.py's
   add_image(), keeping all DB access centralized there. This means
   running ingestion from the CLI (see main() below) extracts image files
   but does not register them in the document_images table — only the
   Streamlit "Run Ingestion" button path in app.py does the full wiring.

Multi-user notes:
- VECTORSTORE_DIR is a single shared persist directory. Chroma supports
  multiple named collections inside one persist directory, so all users'
  vectors live on disk together but are logically isolated by collection.
- DATA_DIR is the base folder; each user's raw uploaded files live under
  DATA_DIR/user_<id>/, kept separate so re-ingestion for one user never
  touches another user's source files.
- Extracted images are saved under DATA_DIR/user_<id>/_images/, i.e.
  inside that same per-user folder — so delete_user_files() (used on
  account deletion) already cleans them up for free, with no extra code
  needed for that case.

Note on PowerPoint files: only .pptx (Office 2007+) is supported.
Legacy .ppt files must be converted to .pptx first (e.g. via LibreOffice)
before placing them in the data folder.

Usage:
    python ingest.py <user_id>          # ingest that user's data folder
    python ingest.py                    # ingests under a "test_user" folder

Requirements:
    pip install langchain langchain-community langchain-ollama chromadb pypdf docx2txt python-pptx
"""

import os
import sys
import shutil
import uuid
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pypdf import PdfReader

# ---------------------- CONFIG ----------------------
DATA_DIR = "data"                # base folder; per-user subfolders live under here
VECTORSTORE_DIR = "vectorstore"  # single shared Chroma persist directory
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
# ------------------------------------------------------


def get_user_data_dir(user_id: int) -> str:
    """Return (and create) the per-user raw-files folder, e.g. data/user_3/."""
    user_dir = os.path.join(DATA_DIR, f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def get_user_images_dir(user_id) -> str:
    """Return (and create) the per-user extracted-images folder, e.g.
    data/user_3/_images/. Lives inside the per-user data folder so account
    deletion's delete_user_files() cleans these up automatically too."""
    images_dir = os.path.join(get_user_data_dir(user_id), "_images")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def get_user_collection_name(user_id: int) -> str:
    """Kept here for convenience; auth.py's AuthManager.get_chroma_collection()
    is the source of truth (it's stored per-user in the DB), but this gives
    you the same deterministic naming if you ever need it without a DB hit."""
    return f"user_{user_id}_docs"


def delete_user_files(user_id) -> None:
    """
    Permanently remove a user's raw uploaded files (their entire per-user
    data subfolder, e.g. data/user_3/). Used when deleting an account.

    This does NOT touch the Chroma vector store — that must be cleared
    separately via the vector store's own delete_collection() API (see
    app.py's clear_vectorstore()), since only the caller holding the live
    connection can do that safely (avoids Windows file-lock issues).
    """
    user_dir = os.path.join(DATA_DIR, f"user_{user_id}")
    if os.path.isdir(user_dir):
        shutil.rmtree(user_dir, ignore_errors=True)


def load_pptx(filepath: str, filename: str):
    """
    Extract text from a .pptx file, one Document per slide.
    Includes shape text and speaker notes. Slide number is stored
    in metadata for precise citation.
    """
    prs = Presentation(filepath)
    docs = []

    for slide_number, slide in enumerate(prs.slides, start=1):
        slide_texts = []

        # Text from all shapes on the slide (titles, bullets, text boxes)
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in paragraph.runs)
                    if text.strip():
                        slide_texts.append(text.strip())

        # Speaker notes, if present
        if slide.has_notes_slide:
            notes_frame = slide.notes_slide.notes_text_frame
            if notes_frame and notes_frame.text.strip():
                slide_texts.append(f"[Speaker notes: {notes_frame.text.strip()}]")

        slide_content = "\n".join(slide_texts)

        if slide_content.strip():
            docs.append(
                Document(
                    page_content=slide_content,
                    metadata={"source": filename, "slide": slide_number},
                )
            )

    return docs


def _soft_mask_ids(page_images) -> set:
    """
    Object ids of images on this page that exist only as another image's
    /SMask (soft mask — the transparency layer for an image, not a picture
    a reader would ever see on its own). Some PDF generators register that
    mask directly in the page's XObject resources, which makes pypdf's
    page.images return it as if it were a second, separate image.
    """
    mask_ids = set()
    for image in page_images:
        ref = getattr(image, "indirect_reference", None)
        if ref is None:
            continue
        obj = ref.get_object()
        if "/SMask" in obj:
            mask_idnum = getattr(obj.raw_get("/SMask"), "idnum", None)
            if mask_idnum is not None:
                mask_ids.add(mask_idnum)
    return mask_ids


def extract_images_from_pdf(filepath: str, filename: str, images_dir: str) -> list:
    """
    Extract embedded images from a PDF, one record per image. Page numbers
    are 0-indexed to exactly match PyPDFLoader's own "page" metadata (set
    on each chunk in load_documents() below), so a retrieved chunk's page
    number can be used directly to look up images from the same page.

    Images that are only another image's soft mask (see _soft_mask_ids)
    are skipped, so they don't show up as a duplicate of the real picture.
    """
    records = []
    try:
        reader = PdfReader(filepath)
        for page_number, page in enumerate(reader.pages):  # 0-indexed, matches PyPDFLoader
            page_images = list(page.images)
            mask_ids = _soft_mask_ids(page_images)
            for image in page_images:
                ref = getattr(image, "indirect_reference", None)
                if ref is not None and ref.idnum in mask_ids:
                    continue  # mask-only object, not a real picture
                ext = os.path.splitext(image.name)[1] or ".png"
                out_name = f"{uuid.uuid4().hex[:12]}{ext}"
                out_path = os.path.join(images_dir, out_name)
                with open(out_path, "wb") as f:
                    f.write(image.data)
                records.append(
                    {"filename": filename, "page_number": page_number, "image_path": out_path}
                )
    except Exception as e:
        print(f"Image extraction failed for {filename}: {e}")
    return records


def extract_images_from_pptx(filepath: str, filename: str, images_dir: str) -> list:
    """
    Extract embedded images from a PPTX, one record per image. Slide
    numbers are 1-indexed to exactly match load_pptx()'s own "slide"
    metadata above, so a retrieved chunk's slide number can be used
    directly to look up images from the same slide.
    """
    records = []
    try:
        prs = Presentation(filepath)
        for slide_number, slide in enumerate(prs.slides, start=1):  # 1-indexed, matches load_pptx()
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image = shape.image
                    ext = f".{image.ext}" if image.ext else ".png"
                    out_name = f"{uuid.uuid4().hex[:12]}{ext}"
                    out_path = os.path.join(images_dir, out_name)
                    with open(out_path, "wb") as f:
                        f.write(image.blob)
                    records.append(
                        {"filename": filename, "page_number": slide_number, "image_path": out_path}
                    )
    except Exception as e:
        print(f"Image extraction failed for {filename}: {e}")
    return records


def extract_images(data_dir: str, user_id) -> list:
    """
    Scan a (per-user) data directory for PDF/PPTX files and extract every
    embedded image, saving each to that user's images folder
    (get_user_images_dir()). Returns a list of
    {"filename", "page_number", "image_path"} records.

    This function only touches the filesystem — it does NOT write to the
    database. The caller (app.py) is responsible for passing each record
    to auth.py's add_image(), keeping all DB access centralized there,
    consistent with how the rest of this module works.
    """
    images_dir = get_user_images_dir(user_id)
    all_records = []

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue

        ext = filename.lower().split(".")[-1]
        if ext == "pdf":
            all_records.extend(extract_images_from_pdf(filepath, filename, images_dir))
        elif ext == "pptx":
            all_records.extend(extract_images_from_pptx(filepath, filename, images_dir))

    return all_records


def load_documents(data_dir: str):
    """Load all supported documents from the given (per-user) data directory."""
    documents = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. Create it and add your files."
        )

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if not os.path.isfile(filepath):
            continue

        ext = filename.lower().split(".")[-1]

        try:
            if ext == "pptx":
                # Handled separately: extracts per-slide text + speaker notes
                loaded_docs = load_pptx(filepath, filename)
            elif ext == "ppt":
                print(
                    f"Skipping '{filename}': legacy .ppt is not supported. "
                    f"Convert it to .pptx first."
                )
                continue
            else:
                if ext == "pdf":
                    loader = PyPDFLoader(filepath)
                elif ext == "docx":
                    loader = Docx2txtLoader(filepath)
                elif ext == "txt":
                    loader = TextLoader(filepath, encoding="utf-8")
                else:
                    print(f"Skipping unsupported file type: {filename}")
                    continue

                loaded_docs = loader.load()

                # Tag each document with its source filename for later citation
                for doc in loaded_docs:
                    doc.metadata["source"] = filename

            documents.extend(loaded_docs)
            print(f"Loaded: {filename} ({len(loaded_docs)} page(s)/slide(s)/section(s))")

        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    if not documents:
        raise ValueError(f"No documents were loaded from '{data_dir}'.")

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def embed_and_store(chunks, persist_dir: str, embed_model: str, collection_name: str):
    """
    Generate embeddings and persist them in a Chroma vector store, scoped to
    the given collection_name so different users' data lives in the same
    persist_dir but never mixes.
    """
    print(
        f"Generating embeddings using '{embed_model}' for collection "
        f"'{collection_name}' (this may take a while)..."
    )

    embeddings = OllamaEmbeddings(model=embed_model)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    print(f"Vector store persisted at: {persist_dir} (collection: {collection_name})")
    return vectorstore


def main():
    # Standalone CLI usage: python ingest.py <user_id>
    user_id = sys.argv[1] if len(sys.argv) > 1 else "test_user"
    data_dir = get_user_data_dir(user_id) if str(user_id).isdigit() else os.path.join(DATA_DIR, str(user_id))
    collection_name = f"user_{user_id}_docs"

    print(f"=== RAG Ingestion Pipeline (user: {user_id}) ===\n")

    print(f"Step 1: Loading documents from '{data_dir}'...")
    documents = load_documents(data_dir)

    print("\nStep 2: Splitting into chunks...")
    chunks = split_documents(documents)

    print("\nStep 3: Embedding + storing in vector database...")
    embed_and_store(chunks, VECTORSTORE_DIR, EMBED_MODEL, collection_name)

    print("\nIngestion complete. You can now run query.py to test retrieval.")


if __name__ == "__main__":
    main()
