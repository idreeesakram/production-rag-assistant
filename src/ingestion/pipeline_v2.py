"""
Ingestion pipeline with token-based chunking (not character-based).
Uses tiktoken to measure chunks in actual tokens instead of characters.
Stores paper_title and paper_id in metadata for later retrieval.
"""

from pathlib import Path
import tiktoken
from loguru import logger

from src.ingestion.parser import extract_pages
from src.ingestion.embedder import embed_chunks
from src.db.chroma_client import upsert_chunks
from src.config import Config


def _tokenize_length(text: str) -> int:
    """Measure text length in tokens using tiktoken (cl100k_base encoding)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception:
        # Fallback: rough estimate of 1 token per 4 characters
        return len(text) // 4


def _chunk_text_by_tokens(
    text: str,
    chunk_size: int = 512,  # tokens, not characters
    overlap: int = 100,  # tokens
) -> list[str]:
    """
    Split text into overlapping chunks, measured by token count.
    
    Args:
        text: Full text to chunk
        chunk_size: Target tokens per chunk (default 512)
        overlap: Token overlap between chunks (default 100)
    
    Returns:
        List of text chunks
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        if chunk_tokens:
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
    
    logger.debug(f"Split {len(tokens)} tokens into {len(chunks)} chunks ({chunk_size}T each, {overlap}T overlap)")
    return chunks


def ingest_v2(pdf_path: str, paper_title: str = None, paper_id: str = None) -> dict:
    """
    Ingest a PDF: parse → chunk by tokens → embed → store.
    
    Args:
        pdf_path: Path to PDF file
        paper_title: Optional full title of the paper (used in metadata and citations)
        paper_id: Optional unique ID for the paper (used in vector DB)
    
    Returns:
        {
            "pages": int,
            "chunks": int,
            "paper_id": str,
            "paper_title": str,
        }
    """
    Config.validate()
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if not paper_title:
        paper_title = pdf_path.stem  # Use filename without extension
    
    if not paper_id:
        paper_id = pdf_path.stem.replace(" ", "-").lower()
    
    # --- Parse PDF ---
    logger.info(f"Parsing PDF: {pdf_path}")
    pages = extract_pages(str(pdf_path))
    total_pages = len(pages)
    
    if not pages:
        raise ValueError(f"No text extracted from PDF: {pdf_path}")
    
    # --- Chunk by tokens ---
    logger.info(f"Chunking {total_pages} pages by tokens ({Config.CHUNK_SIZE}T, {Config.CHUNK_OVERLAP}T overlap)")
    all_chunks = []
    chunk_index = 0
    
    for page_dict in pages:
        page_text = page_dict.get("text", "")
        page_num = page_dict.get("page_num", 1)
        
        if not page_text.strip():
            continue
        
        text_chunks = _chunk_text_by_tokens(
            page_text,
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP,
        )
        
        for chunk_text in text_chunks:
            if chunk_text.strip():
                all_chunks.append({
                    "text": chunk_text,
                    "doc_id": paper_id,
                    "paper_title": paper_title,  # NEW: store full title
                    "page_num": page_num,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
    
    total_chunks = len(all_chunks)
    logger.info(f"Created {total_chunks} token-based chunks")
    
    if not all_chunks:
        raise ValueError(f"No chunks created from PDF: {pdf_path}")
    
    # --- Embed chunks ---
    logger.info(f"Embedding {total_chunks} chunks...")
    embedded_chunks = embed_chunks(all_chunks)
    
    # --- Store in ChromaDB ---
    logger.info(f"Upserting {total_chunks} chunks into ChromaDB...")
    upsert_chunks(embedded_chunks)
    
    logger.info(f"Ingestion complete: {total_pages} pages → {total_chunks} chunks")
    
    return {
        "pages": total_pages,
        "chunks": total_chunks,
        "paper_id": paper_id,
        "paper_title": paper_title,
    }
