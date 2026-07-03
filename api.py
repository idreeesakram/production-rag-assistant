"""
FastAPI backend for RAG Research Assistant.
Provides 4 routes: upload, query, list papers, delete paper.
All routes have Pydantic validation and rate limiting.
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

from src.config import Config
from src.ingestion.pipeline_v2 import ingest_v2 as ingest
from src.db.chroma_client import get_collection, count_chunks, load_all_chunks
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.pipeline import retrieval
from src.generation.related_work import generate_related_work

# --- Rate Limiting ---
rate_limiter: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds


def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    timestamps = rate_limiter[client_ip]
    timestamps[:] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        return False
    timestamps.append(now)
    return True


def _cleanup_old_uploads(upload_dir: Path, max_age_seconds: int = 3600):
    """Delete uploaded PDFs older than max_age_seconds to prevent disk creep."""
    now = time.time()
    for f in upload_dir.glob("*.pdf"):
        try:
            if now - f.stat().st_mtime > max_age_seconds:
                f.unlink()
                logger.info(f"Cleaned up old upload: {f.name}")
        except Exception as e:
            logger.warning(f"Failed to clean up {f.name}: {e}")


# --- Pydantic Models ---
class UploadResponse(BaseModel):
    status: str
    message: str
    file_name: str
    pages: int
    chunks: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    max_retries: int = Field(2, ge=1, le=3)


class QueryResponse(BaseModel):
    status: str
    related_work: str
    cited_papers: list[str]
    total_latency_ms: float
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
    doc_id: str = Field(..., min_length=1)


class DeletePaperResponse(BaseModel):
    status: str
    message: str
    doc_id: str
    chunks_deleted: int


# --- App Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    Config.validate()
    _rebuild_bm25_index()
    yield


app = FastAPI(
    title="RAG Research Assistant API",
    description="Upload papers, query for related work, manage library",
    version="1.0",
    lifespan=lifespan,
)

_bm25_index = None


def _rebuild_bm25_index():
    global _bm25_index
    chunks = load_all_chunks()
    if chunks:
        _bm25_index = build_bm25_index(chunks)
        logger.info(f"BM25 index rebuilt with {len(chunks)} chunks")
    return _bm25_index


# --- Route 1: Upload ---
@app.post("/upload", response_model=UploadResponse)
async def upload_paper(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    safe_filename = Path(file.filename).name
    if not safe_filename or not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_uploads(upload_dir)

        file_path = upload_dir / safe_filename
        with open(file_path, "wb") as f:
            f.write(contents)

        result = ingest(str(file_path))
        _rebuild_bm25_index()

        logger.info(f"Uploaded {safe_filename}: {result['pages']} pages, {result['chunks']} chunks")
        return UploadResponse(
            status="success",
            message=f"Successfully ingested {safe_filename}",
            file_name=safe_filename,
            pages=result["pages"],
            chunks=result["chunks"],
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {str(e)}")


# --- Route 2: Query ---
@app.post("/query", response_model=QueryResponse)
async def query_papers(request: Request, body: QueryRequest):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    if not _bm25_index:
        raise HTTPException(status_code=400, detail="No papers uploaded yet.")

    try:
        t_total_start = time.time()

        # Retrieve once — pass chunks into generation to avoid double retrieval
        t_retrieval_start = time.time()
        chunks = retrieval(body.query, _bm25_index, top_k=10)
        t_retrieval_end = time.time()

        t_generation_start = time.time()
        related_work, cited_papers = generate_related_work(
            query=body.query,
            bm25_index=_bm25_index,
            max_retries=body.max_retries,
            chunks=chunks,  # Pass pre-retrieved chunks — no second retrieval
        )
        t_generation_end = time.time()

        retrieval_ms = (t_retrieval_end - t_retrieval_start) * 1000
        generation_ms = (t_generation_end - t_generation_start) * 1000
        total_ms = (t_generation_end - t_total_start) * 1000

        logger.info(f"Query answered in {total_ms:.0f}ms (retrieval={retrieval_ms:.0f}ms, generation={generation_ms:.0f}ms)")

        return QueryResponse(
            status="success",
            related_work=related_work,
            cited_papers=cited_papers,
            total_latency_ms=round(total_ms, 2),
            retrieval_latency_ms=round(retrieval_ms, 2),
            generation_latency_ms=round(generation_ms, 2),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query generation failed: {str(e)}")


# --- Route 3: List Papers ---
@app.get("/list", response_model=ListPapersResponse)
async def list_papers():
    try:
        collection = get_collection()
        all_docs = collection.get(include=["metadatas"])
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
        return ListPapersResponse(
            status="success",
            papers=papers,
            total_papers=len(papers),
            total_chunks=count_chunks(),
        )
    except Exception as e:
        logger.error(f"List failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list papers: {str(e)}")


# --- Route 4: Delete Paper ---
@app.post("/delete", response_model=DeletePaperResponse)
async def delete_paper(request: Request, body: DeletePaperRequest):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    try:
        collection = get_collection()
        all_docs = collection.get(include=["metadatas"])
        chunk_ids_to_delete = [
            doc_id for i, doc_id in enumerate(all_docs.get("ids", []))
            if all_docs.get("metadatas", [{}])[i].get("doc_id") == body.doc_id
        ]

        if not chunk_ids_to_delete:
            raise HTTPException(status_code=404, detail=f"No paper found with doc_id: {body.doc_id}")

        collection.delete(ids=chunk_ids_to_delete)
        _rebuild_bm25_index()

        logger.info(f"Deleted {len(chunk_ids_to_delete)} chunks for doc_id: {body.doc_id}")
        return DeletePaperResponse(
            status="success",
            message=f"Deleted paper {body.doc_id}",
            doc_id=body.doc_id,
            chunks_deleted=len(chunk_ids_to_delete),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete paper: {str(e)}")


# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
