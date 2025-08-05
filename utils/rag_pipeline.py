# utils/rag_pipeline.py
"""
RAG pipeline utilities for the GitHub + PDF analyser.

Fix history
-----------
✓ Wait for Pinecone deletions & 409 retry
✓ Robust cleanup_old_indexes
✓ Build VectorStore from explicit Index (fix 401 + kwargs issue)
✓ **NEW**: Uses retriever.invoke() (no deprecation warnings)
✓ **NEW**: Silences HuggingFace tokenizers fork warning
"""
from __future__ import annotations

import os
import time
import uuid
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from pinecone import Pinecone, ServerlessSpec
from pinecone.openapi_support.exceptions import PineconeApiException

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from transformers import pipeline

# --------------------------------------------------------------------------- #
#  One‑liner to silence HF tokenizers fork warning                            #
# --------------------------------------------------------------------------- #
def _set_env_silently(key: str, value: str):
    if os.environ.get(key) is None:
        os.environ[key] = value

_set_env_silently("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def cleanup_old_indexes(pc: Pinecone, prefix: str = "repo-pdf-rag-") -> None:
    """Delete every Pinecone index whose name starts with *prefix*."""
    for item in pc.list_indexes():
        if isinstance(item, str):
            idx_name = item
        elif isinstance(item, dict):
            idx_name = item.get("name")
        else:
            idx_name = getattr(item, "name", None)
        if isinstance(idx_name, str) and idx_name.startswith(prefix):
            pc.delete_index(idx_name)


def _wait_until_deleted(pc: Pinecone, name: str, timeout: int = 60) -> None:
    """Poll until *name* disappears from list_indexes() or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = [
            i["name"] if isinstance(i, dict) else getattr(i, "name", i)
            for i in pc.list_indexes()
        ]
        if name not in current:
            return
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for Pinecone to delete index {name!r}")


def gemini_answer(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    return model.generate_content(prompt).text


# --------------------------------------------------------------------------- #
#  RAG Chain                                                                  #
# --------------------------------------------------------------------------- #
def setup_rag_chain(documents, *, index_name: str | None = "repo-pdf-rag"):
    if index_name is None:
        index_name = f"repo-pdf-rag-{uuid.uuid4().hex[:8]}"

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found.")
    pc = Pinecone(api_key=api_key)

    # 1) Delete same‑named index if present and wait
    existing = [
        i["name"] if isinstance(i, dict) else getattr(i, "name", i)
        for i in pc.list_indexes()
    ]
    if index_name in existing:
        pc.delete_index(index_name)
        _wait_until_deleted(pc, index_name)

    # 2) Clean up old auto‑generated indexes
    cleanup_old_indexes(pc, prefix="repo-pdf-rag-")

    # 3) Create new index (retry once on 409)
    for attempt in (1, 2):
        try:
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            break
        except PineconeApiException as e:
            if e.status == 409 and attempt == 1:
                time.sleep(5)
                _wait_until_deleted(pc, index_name, timeout=30)
                continue
            raise

    # 4) Build vectorstore from explicit Index object
    index = pc.Index(index_name)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = PineconeVectorStore(index=index, embedding=embed)
    vector_store.add_documents(documents)

    return vector_store


# --------------------------------------------------------------------------- #
#  Local summariser fallback                                                  #
# --------------------------------------------------------------------------- #
CODE_MODEL = "Salesforce/codet5-small"
SUMMARIZER = pipeline("summarization", model=CODE_MODEL, tokenizer=CODE_MODEL)


def run_query(vectorstore, query: str, k: int = 5) -> str:
    """Retrieve context from *vectorstore* and answer *query*."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)  # <— no deprecation warning

    context = ""
    for d in docs:
        path = d.metadata.get("path", "unknown")
        context += f"\n\n--- File: {path} ---\n{d.page_content}"

    # --- Gemini first ----------------------------------------------------- #
    if os.getenv("GOOGLE_API_KEY"):
        prompt = (
            "You are a senior software engineer and code reviewer.\n\n"
            "Analyze the following extracted files (only .py and .md) and "
            "produce a comprehensive report:\n\n"
            "1. **Repository Overview**\n"
            "   - Name & purpose\n"
            "   - High-level functionality and domain context\n\n"
            "2. **File Summaries**\n"
            "   • **Path** – relative location\n"
            "   • **Purpose** – 1–2 sentence summary\n"
            "   • **Key Components** – main classes/functions\n"
            "   • **Dependencies** – external libraries/services\n\n"
            "3. **Architecture & Flow**\n"
            "   - Module interactions and data/control flow\n"
            "   - Observed design patterns or architectural styles\n\n"
            "4. **Usage & Setup**\n"
            "   - Installation and configuration steps\n"
            "   - How to run key entry points or scripts\n\n"
            "5. **Strengths & Suggestions**\n"
            "   - What the codebase does well\n"
            "   - Recommendations for refactoring, testing, or enhancements\n\n"
            f"6. **Answer to the user’s query**: \"{query}\"\n\n"
            "Now review these file contents:\n"
            f"{context}"
        )
        try:
            return gemini_answer(prompt)
        except ResourceExhausted:
            print("⚠️  Gemini quota exhausted; falling back to local summariser.")

    # --- Local fallback --------------------------------------------------- #
    chunks = [context[i : i + 800] for i in range(0, len(context), 800)]
    partials = [
        SUMMARIZER(
            f"Summarise this code:\n{chunk}",
            max_length=150,
            min_length=50,
            do_sample=False,
        )[0]["summary_text"].strip()
        for chunk in chunks
    ]
    combined = " ".join(partials)
    final = SUMMARIZER(
        f"Combine and refine these summaries:\n{combined}",
        max_length=200,
        min_length=100,
        do_sample=False,
    )[0]["summary_text"].strip()
    return final
