# Production RAG Assistant

A Streamlit-based AI assistant for uploading research papers and asking questions grounded in the source documents. It uses retrieval-augmented generation to provide answers with citations and page references, making it useful for researchers, students, and anyone working with technical papers.

## Features

- Upload PDF research papers
- Ask questions in natural language
- Get answers grounded in the uploaded documents
- View citations and source snippets
- Use Groq, Anthropic, or OpenAI as the LLM provider
- Run locally or deploy to Streamlit Cloud

## Quick Start

1. Clone the repository
2. Create and activate a virtual environment
3. Install the dependencies
4. Add your API key to a `.env` file or Streamlit secrets
5. Run the app

```bash
git clone https://github.com/idreeesakram/production-rag-assistant.git
cd production-rag-assistant
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables

Set at least one of the following:

- `GROQ_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

Optional settings:

- `GROQ_MODEL`
- `ANTHROPIC_MODEL`
- `OPENAI_MODEL`

## Project Structure

- `app.py` — main Streamlit app
- `src/config.py` — configuration and secret handling
- `src/ingestion/` — PDF parsing and chunking
- `src/retrieval/` — search and reranking logic
- `src/generation/` — LLM answer generation

## Deployment

For Streamlit Cloud deployment, add your API key in the app secrets section using the same variable name as above.

## License

This project is for personal and educational use.


| Layer | Technology |
|---|---|
| PDF Parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Sparse Retrieval | BM25Retriever |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq / Anthropic / OpenAI (with retry + failover) |
| Orchestration | LangChain |
| UI | Streamlit |
| Observability | Langfuse |
| Evaluation | Ragas |
| CI/CD | GitHub Actions |

---

# Local Setup

<details>
<summary><b>Setup Instructions</b></summary>

<br>

```bash
# Clone repository
git clone https://github.com/aieng-abdullah/production-rag-assistant.git

# Move into project
cd production-rag-assistant

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

# Add GROQ_API_KEY inside .env

# Start application
streamlit run app.py
```

</details>

---

# Running Evaluation

```bash
python3 eval/eval_runner.py
```

Evaluation results are saved to `results.json`.

---

# Running Tests

```bash
pytest tests/ -v
```

---

# Performance Notes

End-to-end latency ranges from 1.54s (p50) to 14.06s (p99) across 141 traced requests.

The cross-encoder reranker accounts for 72% of total runtime on CPU with 12x variance between p50 and p95.

Current implementation prioritizes retrieval quality and grounded answers over raw latency.

---

