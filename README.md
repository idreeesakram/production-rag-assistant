# Production RAG Assistant

A Streamlit-based retrieval-augmented generation assistant for PDF research papers. Upload papers, ask research questions, and receive answers grounded in the source documents.

## Features

- Upload PDF research papers
- Extract document chunks and build a BM25 index
- Generate cited related work summaries using a language model
- Support for Groq, Anthropic, and OpenAI providers
- Optional provider override via the Streamlit sidebar
- Local evaluation runner and test suite

## Quick Start

1. Clone the repository
2. Create and activate a Python virtual environment
3. Install dependencies
4. Create a `.env` file with at least one LLM provider API key
5. Run the app

```bash
git clone https://github.com/idreeesakram/production-rag-assistant.git
cd production-rag-assistant
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Create a .env file with one key, for example:
# GROQ_API_KEY=your_groq_api_key
streamlit run app.py
```

## Environment Variables

At least one of these must be set:

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

Dependencies are installed from `requirements.txt`, including `streamlit` and `python-dotenv`.

The app loads `.env` automatically via `python-dotenv`, and also supports Streamlit secrets in deployment.

## Project Structure

- `app.py` — main Streamlit application
- `src/config.py` — environment and provider config
- `src/ingestion/` — PDF parsing and chunking pipeline
- `src/retrieval/` — BM25 indexing and hybrid search
- `src/generation/` — related work generation and answer assembly
- `src/db/` — ChromaDB helper functions
- `tests/` — pytest suite
- `eval/` — evaluation runner

---

## Running Evaluation

```bash
python eval/eval_runner.py
```

Evaluation results are saved to `results.json`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

