"""Retrieval-Augmented Generation (RAG) index management and similarity search via FAISS."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    GEMINI_API_KEY,
    KNOWLEDGE_DIR,
)

logger = logging.getLogger(__name__)


def _create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini embeddings client configured from environment variables."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


def _load_knowledge_documents(knowledge_dir: Path) -> list[Document]:
    """Load all .md and .txt files from the knowledge directory."""
    if not knowledge_dir.exists():
        logger.warning("Knowledge directory does not exist: %s", knowledge_dir)
        return []

    documents: list[Document] = []
    for pattern in ("**/*.md", "**/*.txt"):
        loader = DirectoryLoader(
            str(knowledge_dir),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        documents.extend(loader.load())
    return documents


def build_index(
    knowledge_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
) -> FAISS | None:
    """Load knowledge docs, chunk, embed with Gemini, and persist a local FAISS index."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is missing; cannot build FAISS embeddings index.")
        return None

    source_dir = Path(knowledge_dir) if knowledge_dir is not None else KNOWLEDGE_DIR
    target_dir = Path(index_dir) if index_dir is not None else FAISS_INDEX_DIR

    try:
        docs = _load_knowledge_documents(source_dir)
        if not docs:
            logger.warning("No .md or .txt documents found in %s.", source_dir)
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            logger.warning("Document splitting produced zero chunks from %s.", source_dir)
            return None

        embeddings = _create_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        target_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(target_dir))
        logger.info("Built FAISS index with %d chunks at %s.", len(chunks), target_dir)
        return vectorstore
    except Exception:
        logger.exception("Failed to build FAISS index from %s.", source_dir)
        return None


def load_index(
    knowledge_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
) -> FAISS | None:
    """Load an existing local FAISS index or build one if it does not exist yet."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is missing; cannot load FAISS index.")
        return None

    target_dir = Path(index_dir) if index_dir is not None else FAISS_INDEX_DIR
    index_file = target_dir / "index.faiss"
    store_file = target_dir / "index.pkl"

    if index_file.exists() and store_file.exists():
        try:
            embeddings = _create_embeddings()
            # Safe here because index.pkl is generated strictly from our local
            # data/knowledge files via build_index(), never from untrusted uploads.
            return FAISS.load_local(
                str(target_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            logger.exception("Failed to load existing FAISS index; attempting rebuild.")

    return build_index(knowledge_dir=knowledge_dir, index_dir=target_dir)


def retrieve(
    query: str,
    k: int = 3,
    knowledge_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
) -> list[tuple[str, str, float]]:
    """Retrieve top-k matching chunks as (text, source_name, score) tuples."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return []

    vectorstore = load_index(knowledge_dir=knowledge_dir, index_dir=index_dir)
    if vectorstore is None:
        return []

    try:
        matches = vectorstore.similarity_search_with_score(cleaned_query, k=k)
        results: list[tuple[str, str, float]] = []
        for doc, score in matches:
            raw_source = str(doc.metadata.get("source", "unknown"))
            source_name = Path(raw_source).name
            results.append((doc.page_content.strip(), source_name, float(score)))
        return results
    except Exception:
        logger.exception("FAISS similarity search failed for query='%s'.", cleaned_query)
        return []


def format_context(results: list[tuple[str, str, float]]) -> str:
    """Format retrieved (text, source, score) hits into a context string with source names."""
    if not results:
        return ""

    blocks: list[str] = []
    for text, source, score in results:
        blocks.append(f"[Source: {source} | distance={score:.4f}]\n{text}")
    return "\n\n".join(blocks)
