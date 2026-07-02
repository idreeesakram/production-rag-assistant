"""
ChromaDB client using pure LangChain Chroma integration.
"""
from typing import Optional, List, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from src.config import Config
from src.ingestion.embedder import _get_model as get_embedding_model

_vectorstore: Optional[Chroma] = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embedding_model = get_embedding_model()
        _vectorstore = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=str(Config.CHROMA_DIR),
        )
        logger.info(f"LangChain Chroma initialized: {Config.COLLECTION_NAME}")
    return _vectorstore


def get_collection():
    vectorstore = _get_vectorstore()
    return vectorstore._collection


def upsert_chunks(chunks: List[Dict]) -> int:
    vectorstore = _get_vectorstore()
    documents = []
    ids = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "doc_id": chunk.get("doc_id", "unknown"),
                "page_num": chunk.get("page_num", -1),
                "chunk_index": chunk.get("chunk_index", -1),
            }
        )
        documents.append(doc)
        doc_id = chunk.get("doc_id", "unknown")
        chunk_index = chunk.get("chunk_index", len(ids))
        ids.append(chunk.get("chunk_id", f"{doc_id}_chunk_{chunk_index}"))
    vectorstore.add_documents(documents=documents, ids=ids)
    logger.info(f"Upserted {len(chunks)} chunks to vectorstore")
    return len(chunks)


def load_all_chunks() -> List[Dict]:
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection
    results = collection.get()
    chunks = []
    for text, metadata in zip(results["documents"], results["metadatas"]):
        doc_id = metadata.get("doc_id", "unknown")
        chunk_index = metadata.get("chunk_index", 0)
        chunks.append({
            "text": text,
            "chunk_id": f"{doc_id}_chunk_{chunk_index}",
            **metadata
        })
    return chunks


def has_chunks() -> bool:
    return count_chunks() > 0


def count_chunks() -> int:
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection
    return collection.count()


def reset_client():
    global _vectorstore
    _vectorstore = None
    logger.debug("Vectorstore reset")


def get_vectorstore() -> Chroma:
    return _get_vectorstore()


def get_all_paper_titles() -> List[str]:
    """Return a deduplicated list of paper titles from all stored chunks."""
    if not has_chunks():
        return []
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection
    results = collection.get(include=["metadatas"])
    seen = set()
    titles = []
    for metadata in results["metadatas"]:
        title = metadata.get("doc_id", "")
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles
