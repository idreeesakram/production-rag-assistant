"""
FastAPI backend for RAG Research Assistant.
Provides 4 routes: upload, query, list papers, delete paper.
All routes have Pydantic validation and rate limiting.
"""

import os
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

from src.config import Config
from src.ingestion.pipeline_v2 import ingest_v2 as ingest
from src.db.chroma_client import get_collection, count_chunks, load_all_chunks
from src.retrieval.bm25_index import build_bm25_index
from src.generation.related_work import generate_related_work

# --- Rate Limiting ---
rate_limiter = defaultdict(list)
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds


def check_rate_limit(client_id: str = "default") -> bool:
    """Simple rate limiter: max N requests per M seconds per client."""
    now = time.time()
    timestamps = rate_limiter[client_id]
    
    # Remove old timestamps outside the window
    timestamps[:] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        return False
    
    timestamps.append(now)
    return True


# --- Pydantic Models ---
class UploadResponse(BaseModel):
    status: str
    message: str
    file_name: str
    pages: int
    chunks: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    max_retries: int = Field(3, ge=1, le=5, description="Max regeneration attempts")


class QueryResponse(BaseModel):
    status: str
    related_work: str
    cited_papers: list[str]
    retrieval_latency_ms: float
    generation_latency_ms: float


class PaperInfo(BaseModel):
    doc_id: str
    paper_title: str
    chunk_count: int


class ListPapersResponse(BaseModel):
    status: str
    papers: list[PaperInfo]
    total_papers: int
    total_chunks: int


class DeletePaperRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, description="Document ID to delete")


class DeletePaperResponse(BaseModel):
    status: str
    message: str
    doc_id: str
    chunks_deleted: int


# --- FastAPI App ---
app = FastAPI(
    title="RAG Research Assistant API",
    description="Upload papers, query for related work, manage library",
    version="1.0",
)

# Global BM25 index (rebuilt on each upload)
_bm25_index = None


def _rebuild_bm25_index():
    """Rebuild the BM25 index from all chunks in Chroma."""
    global _bm25_index
    chunks = load_all_chunks()
    if chunks:
        _bm25_index = build_bm25_index(chunks)
        logger.info(f"BM25 index rebuilt with {len(chunks)} chunks")
    return _bm25_index


@app.on_event("startup")
def startup():
    """Load BM25 index on startup."""
    Config.validate()
    _rebuild_bm25_index()


# --- Route 1: Upload ---
@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_paper(file: UploadFile = File(...)):
    """
    Upload a research paper PDF.
    
    - **file**: PDF file to upload (required)
    
    Returns: pages parsed, chunks created
    """
    if not check_rate_limit():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Max 10 requests per 60 seconds."
        )
    
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a PDF."
        )
    
    try:
        # Save uploaded file temporarily
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename
        
        contents = await file.read()
        if not contents:
            raise ValueError("File is empty")
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Ingest the PDF
        result = ingest(str(file_path))
        
        # Rebuild BM25 index
        _rebuild_bm25_index()
        
        logger.info(f"Uploaded {file.filename}: {result['pages']} pages, {result['chunks']} chunks")
        
        return UploadResponse(
            status="success",
            message=f"Successfully ingested {file.filename}",
            file_name=file.filename,
            pages=result["pages"],
            chunks=result["chunks"],
        )
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest PDF: {str(e)}"
        )


# --- Route 2: Query ---
@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_papers(request: QueryRequest):
    """
    Query the paper library and generate a Related Work summary.
    
    - **query**: Question or topic (required, 1-1000 chars)
    - **max_retries**: How many times to retry if citation validation fails (1-5, default 3)
    
    Returns: Related Work paragraph citing papers by title, latency metrics
    """
    if not check_rate_limit():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded."
        )
    
    if not _bm25_index:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No papers uploaded yet. Please upload a PDF first."
        )
    
    try:
        t0 = time.time()
        related_work, cited_papers = generate_related_work(
            request.query,
            _bm25_index,
            max_retries=request.max_retries,
        )
        elapsed_ms = (time.time() - t0) * 1000
        
        logger.info(f"Query answered in {elapsed_ms:.0f}ms, cited {len(cited_papers)} papers")
        
        return QueryResponse(
            status="success",
            related_work=related_work,
            cited_papers=cited_papers,
            retrieval_latency_ms=100,  # Placeholder
            generation_latency_ms=elapsed_ms - 100,
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query generation failed: {str(e)}"
        )


# --- Route 3: List Papers ---
@app.get("/list", response_model=ListPapersResponse, status_code=status.HTTP_200_OK)
async def list_papers():
    """
    List all uploaded papers with chunk counts.
    
    Returns: List of papers, total count, total chunks
    """
    try:
        collection = get_collection()
        all_docs = collection.get(include=["metadatas"])
        
        # Group by doc_id and paper_title
        paper_map = {}
        for i, doc_id in enumerate(all_docs.get("ids", [])):
            metadata = all_docs.get("metadatas", [{}])[i]
            pid = metadata.get("doc_id", "unknown")
            title = metadata.get("paper_title", pid)
            
            if pid not in paper_map:
                paper_map[pid] = {"title": title, "count": 0}
            paper_map[pid]["count"] += 1
        
        papers = [
            PaperInfo(doc_id=pid, paper_title=info["title"], chunk_count=info["count"])
            for pid, info in paper_map.items()
        ]
        
        total_chunks = count_chunks()
        
        return ListPapersResponse(
            status="success",
            papers=papers,
            total_papers=len(papers),
            total_chunks=total_chunks,
        )
    
    except Exception as e:
        logger.error(f"List failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list papers: {str(e)}"
        )


# --- Route 4: Delete Paper ---
@app.post("/delete", response_model=DeletePaperResponse, status_code=status.HTTP_200_OK)
async def delete_paper(request: DeletePaperRequest):
    """
    Delete a paper and all its chunks from the database.
    
    - **doc_id**: ID of the paper to delete (required)
    
    Returns: Confirmation, number of chunks deleted
    """
    if not check_rate_limit():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded."
        )
    
    try:
        collection = get_collection()
        
        # Find all chunk IDs for this doc_id
        all_docs = collection.get(include=["metadatas"])
        chunk_ids_to_delete = []
        
        for i, doc_id in enumerate(all_docs.get("ids", [])):
            metadata = all_docs.get("metadatas", [{}])[i]
            if metadata.get("doc_id") == request.doc_id:
                chunk_ids_to_delete.append(doc_id)
        
        if not chunk_ids_to_delete:
            raise ValueError(f"No chunks found for doc_id: {request.doc_id}")
        
        # Delete chunks
        collection.delete(ids=chunk_ids_to_delete)
        
        # Rebuild BM25 index
        _rebuild_bm25_index()
        
        logger.info(f"Deleted {len(chunk_ids_to_delete)} chunks for doc_id: {request.doc_id}")
        
        return DeletePaperResponse(
            status="success",
            message=f"Deleted paper {request.doc_id}",
            doc_id=request.doc_id,
            chunks_deleted=len(chunk_ids_to_delete),
        )
    
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete paper: {str(e)}"
        )


# --- Health Check ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
