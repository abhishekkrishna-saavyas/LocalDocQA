"""
query.py
---------
Core RAG engine: retrieves relevant chunks from the persisted Chroma
vector store and feeds them to Llama 3.2 (via Ollama) to generate an
answer, with source citations and conversational memory.

Multi-user note: VECTORSTORE_DIR is a single shared persist directory,
but every function that touches Chroma now takes a collection_name so
each user only ever reads/writes their own collection
(f"user_{user_id}_docs") within it.

Supports two answer modes:
- strict: answer ONLY from retrieved document context (default, safest)
- hybrid: prioritize document context, but allow the LLM to supplement
          with its own general knowledge when the context falls short,
          clearly labeling that portion as such

Conversational memory: follow-up questions (e.g. "more details", "why?")
are rewritten into standalone questions using recent chat history before
retrieval, so the retriever searches for the right thing even when the
question alone is ambiguous. That history is now expected to come from
a specific conversation (see auth.py's get_messages), not just a flat
session list, so memory stays scoped per named chat as well as per user.

Usage:
    python query.py <user_id> "your question here"
    python query.py <user_id> --hybrid "your question here"

    Or run without a question for an interactive loop (keeps memory
    across turns within that session):
    python query.py <user_id>
    python query.py <user_id> --hybrid

Requirements:
    pip install langchain langchain-community langchain-ollama chromadb
"""

import sys
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------- CONFIG ----------------------
VECTORSTORE_DIR = "vectorstore"  # single shared Chroma persist directory
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
# LLM_MODEL = "qwen2.5"
TOP_K = 4                      # default; can be overridden per-call via build_qa_engine
DEFAULT_TEMPERATURE = 0.2      # default; can be overridden per-call via build_qa_engine
# Max distance (lower = closer match) a retrieved chunk can have and still be
# allowed to pull in its page's images. Only affects which images are shown —
# text retrieval still returns the top-K chunks regardless, so the answer
# itself is unaffected. Chroma's default metric is squared L2 distance, so the
# right value depends on your embeddings: raise it if relevant images stop
# appearing, lower it if unrelated ones still slip through.
IMAGE_MATCH_MAX_DISTANCE = 1.0
MAX_HISTORY_TURNS = 6  # how many recent user/assistant turns to keep as context
# ------------------------------------------------------

CONTEXTUALIZE_TEMPLATE = """Given the conversation history and a follow-up question, rewrite the \
follow-up question into a standalone question that includes any necessary context from the \
history (e.g. resolve "it", "that", "more details" into what they refer to).

Only output the rewritten standalone question. Do not answer it. If the follow-up question is \
already standalone and needs no context, return it unchanged.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""

STRICT_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based only on the provided document context.

Conversation history (for understanding what the user is referring to only — do not pull facts from it):
{chat_history}

Rules:
- Use ONLY the information in the context below to answer the current question.
- If the answer is not in the context, say "I don't have enough information in the documents to answer that."
- Do not make up information that isn't supported by the context.
- Keep answers clear and concise.

Context:
{context}

Question: {question}

Answer:"""

HYBRID_PROMPT_TEMPLATE = """You are a helpful assistant answering questions using the provided document context as your primary source.

Conversation history (for understanding what the user is referring to, and for staying on the same topic):
{chat_history}

Rules:
- Prioritize the context below when answering. If it fully answers the question, answer from it alone.
- If the context is missing, empty, or only partially answers the question, supplement with your own general knowledge about the same topic to give a complete, useful answer.
- Clearly separate the two: label any part that comes from your own knowledge (not the documents) with the prefix "[General knowledge]".
- Never blend fabricated details into the parts attributed to the documents.
- Keep answers clear and concise.

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore(collection_name: str):
    """Load the persisted Chroma vector store, scoped to one user's collection."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    count = vectorstore._collection.count()
    if count == 0:
        raise ValueError(
            f"Collection '{collection_name}' is empty. Run ingest.py for this user first."
        )

    return vectorstore


def format_docs(docs):
    """Format retrieved chunks into a single context string, tagged by source."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        slide = doc.metadata.get("slide")
        tag = f"{source} (slide {slide})" if slide else source
        formatted.append(f"[Source: {tag}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def format_chat_history(chat_history):
    """
    Format recent chat history into plain text for the prompts.
    chat_history: list of {"role": "user"|"assistant", "content": str}
    """
    if not chat_history:
        return "(no previous conversation)"

    recent = chat_history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


def build_qa_engine(
    vectorstore,
    mode: str = "strict",
    top_k: int = TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
):
    """
    Build a conversational QA function backed by the given (already
    user-scoped) vector store.

    mode: "strict" (default) answers only from document context.
          "hybrid" allows the LLM to supplement with general knowledge,
          clearly labeled, when the context doesn't fully answer.
    top_k: number of chunks retrieved from the vector store per question.
           Higher = more context but more noise/slower; lower = more
           focused but may miss relevant material. Defaults to TOP_K.
    temperature: LLM generation randomness (0.0 = deterministic/focused,
           higher = more varied/creative). Defaults to DEFAULT_TEMPERATURE.

    Returns:
        ask(question: str, chat_history: list) -> (answer: str, source_docs: list)

    chat_history should be the messages for the CURRENT conversation only
    (e.g. from auth.get_messages(conversation_id, user_id)), not every
    conversation the user has ever had — that keeps follow-up resolution
    and memory scoped to one named chat, matching the ChatGPT-style UI.
    """
    if mode not in ("strict", "hybrid"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'strict' or 'hybrid'.")

    llm = ChatOllama(model=LLM_MODEL, temperature=temperature)

    contextualize_prompt = ChatPromptTemplate.from_template(CONTEXTUALIZE_TEMPLATE)
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    answer_template = STRICT_PROMPT_TEMPLATE if mode == "strict" else HYBRID_PROMPT_TEMPLATE
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    answer_chain = answer_prompt | llm | StrOutputParser()

    def ask(question: str, chat_history=None):
        chat_history = chat_history or []
        history_text = format_chat_history(chat_history)

        # Step 1: resolve follow-ups ("more details", "why?") into a standalone
        # question so retrieval searches for the right thing.
        if chat_history:
            standalone_question = contextualize_chain.invoke(
                {"chat_history": history_text, "question": question}
            ).strip()
        else:
            standalone_question = question

        # Step 2: retrieve using the standalone question. Same top-K chunks a
        # plain retriever would return, but each chunk's distance score is
        # attached to its metadata so callers can judge how close the match
        # really was (retrieval always returns K chunks, relevant or not).
        docs = []
        for doc, score in vectorstore.similarity_search_with_score(standalone_question, k=top_k):
            doc.metadata["score"] = score
            docs.append(doc)
        context = format_docs(docs)

        # Step 3: generate the answer, grounded in context, aware of conversation
        answer = answer_chain.invoke(
            {"context": context, "question": question, "chat_history": history_text}
        )

        return answer, docs

    return ask


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python query.py <user_id> [--hybrid] [\"question\"]")
        return

    user_id = args.pop(0)
    collection_name = f"user_{user_id}_docs"

    mode = "strict"
    if "--hybrid" in args:
        mode = "hybrid"
        args.remove("--hybrid")

    print("=== RAG Query Engine ===")
    print(
        f"User: {user_id} | LLM: {LLM_MODEL} | Embeddings: {EMBED_MODEL} "
        f"| Top-K: {TOP_K} | Mode: {mode}\n"
    )

    vectorstore = load_vectorstore(collection_name)
    qa = build_qa_engine(vectorstore, mode=mode)
    chat_history = []

    def run_turn(question: str):
        print(f"\nQuestion: {question}")
        print("Thinking...\n")

        answer, docs = qa(question, chat_history)

        print(f"Answer:\n{answer}\n")

        sources = sorted(set(doc.metadata.get("source", "unknown") for doc in docs))
        print(f"Sources used: {', '.join(sources)}")
        print("-" * 60)

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

    # Single question passed as a command-line argument
    if args:
        run_turn(" ".join(args))
        return

    # Interactive mode — memory persists across turns in this session
    print("Interactive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if not question:
            continue
        run_turn(question)


if __name__ == "__main__":
    main()
