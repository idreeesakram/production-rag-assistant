# Production RAG Research Citation Assistant

A production-grade Retrieval-Augmented Generation (RAG) system for academic researchers. Upload a library of PDF research papers, then generate grounded 2–3 paragraph Related Work summaries that cite papers by title — not by chunk index.

**Live deployment:** (https://production-rag-assistant-production.up.railway.app/)

## Features

- Upload multiple PDF research papers into a persistent vector library
- Token-based chunking (512 tokens, 100 overlap) via tiktoken cl100k_base
- Hybrid retrieval: BM25 + vector search fused with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) over top 20 candidates
- Generates 2–3 paragraph Related Work summaries citing papers by title
- Citation validation with regeneration loop (up to 3 retries via Pydantic)
- FastAPI backend with 4 routes: upload, query, list papers, delete paper
- Multi-provider LLM support: Groq → Anthropic → OpenAI with failover
- Langfuse observability tracing
- ChromaDB persisted to disk — data survives server restarts

## Quick Start

```bash
git clone https://github.com/idreeesakram/production-rag-assistant.git
cd production-rag-assistant
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
# Create a .env file with at least one LLM provider key
streamlit run app.py
```

To run the FastAPI backend alongside the UI:

```bash
uvicorn src.api:app --reload --port 8001
```

## API Routes

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/upload` | Upload a PDF and ingest into vector DB |
| POST | `/query` | Generate a Related Work summary from the library |
| GET | `/papers` | List all stored papers |
| DELETE | `/papers/{paper_id}` | Delete a paper and all its chunks |

All routes use Pydantic validation — missing or invalid fields return HTTP 422.

## Environment Variables

At least one LLM provider key must be set:

- `GROQ_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

Optional settings:

- `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)
- `ANTHROPIC_MODEL` (default: `claude-sonnet-4-20250514`)
- `OPENAI_MODEL` (default: `gpt-4o`)
- `CHROMA_HOST` (default: `localhost`)
- `CHROMA_PORT` (default: `8000`)
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (default: `https://cloud.langfuse.com`)
- `LOG_LEVEL` (default: `INFO`)

## Project Structure

- `app.py` — Streamlit UI
- `src/api.py` — FastAPI backend (4 routes)
- `src/config.py` — environment and provider config
- `src/pipeline_v2.py` — token-based ingestion pipeline (tiktoken)
- `src/related_work.py` — Related Work generation with citation regeneration loop
- `src/ingestion/` — PDF parsing and chunking
- `src/retrieval/` — BM25 indexing and hybrid search
- `src/generation/` — answer assembly
- `src/db/` — ChromaDB helpers
- `tests/` — pytest suite (140 tests, 68% coverage)
- `eval/` — evaluation runner

## Evaluation

```bash
python eval/eval_runner.py
```

Evaluated on 5 question-answer pairs using Ragas (Groq Llama 3.3 70B as judge):

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Faithfulness | 0.83 | 0.75 | PASS |
| Answer Relevancy | 0.90 | 0.75 | PASS |
| Context Recall | 1.00 | 0.70 | PASS |

Faithfulness of 0.83 means 83% of generated claims are grounded in retrieved context. Results are based on a small evaluation set (n=5) and should be treated as indicative, not definitive.

Evaluation results are saved to `results.json`.

## Observability

End-to-end request tracing via Langfuse. 141 traces collected during development and testing.

The cross-encoder reranker is the primary latency driver (~72% of total runtime). Groq LLM generation at p50 is 0.57s.

## Running Tests

```bash
pytest tests/ -v
```
