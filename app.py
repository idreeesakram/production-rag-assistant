"""Streamlit UI for RAG Research Assistant."""

import os
import shutil
from pathlib import Path

import streamlit as st
from loguru import logger

from src.config import Config
from src.db.chroma_client import has_chunks, load_all_chunks, count_chunks
from src.generation.related_work import generate_related_work
from src.ingestion.pipeline_v2 import ingest_v2 as ingest
from src.retrieval.bm25_index import build_bm25_index

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide",
)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    if "ingested_docs" not in st.session_state:
        st.session_state.ingested_docs = []
    if "anthropic_key" not in st.session_state:
        st.session_state.anthropic_key = ""
    if "anthropic_model" not in st.session_state:
        st.session_state.anthropic_model = "claude-sonnet-4-20250514"
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4o"


def save_uploaded_file(uploaded_file) -> Path:
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_pdf(file_path: Path):
    with st.spinner(f"Processing {file_path.name}..."):
        progress_bar = st.progress(0)
        try:
            result = ingest(str(file_path), paper_title=file_path.stem)
            progress_bar.progress(50)
            chunks = load_all_chunks()
            progress_bar.progress(75)
            if chunks:
                st.session_state.bm25_index = build_bm25_index(chunks)
            progress_bar.progress(100)
            if file_path.name not in st.session_state.ingested_docs:
                st.session_state.ingested_docs.append(file_path.name)
            st.success(f"✅ Processed {result['pages']} pages, {result['chunks']} chunks")
            logger.info(f"PDF processed: {file_path.name}")
        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")
            logger.error(f"PDF processing failed: {e}")
            raise


def _apply_provider_overrides():
    if st.session_state.get("anthropic_key"):
        Config.ANTHROPIC_API_KEY = st.session_state.anthropic_key
    if st.session_state.get("anthropic_model"):
        Config.ANTHROPIC_MODEL = st.session_state.anthropic_model
    if st.session_state.get("openai_key"):
        Config.OPENAI_API_KEY = st.session_state.openai_key
    if st.session_state.get("openai_model"):
        Config.OPENAI_MODEL = st.session_state.openai_model


def handle_query(query: str):
    if not has_chunks():
        st.warning("⚠️ Please upload a PDF first!")
        return
    if st.session_state.bm25_index is None:
        st.warning("⚠️ BM25 index not ready. Please process a PDF first.")
        return

    _apply_provider_overrides()
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant papers and generating Related Work..."):
            try:
                related_work_text, cited_papers = generate_related_work(
                    query=query,
                    bm25_index=st.session_state.bm25_index,
                )
                st.markdown("### Related Work")
                st.markdown(related_work_text)
                if cited_papers:
                    st.markdown("---")
                    st.markdown("**Papers Cited:**")
                    for title in cited_papers:
                        st.markdown(f"- {title}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": related_work_text,
                    "cited_papers": cited_papers,
                })
            except Exception as e:
                st.error(f"❌ Error generating Related Work: {e}")
                logger.error(f"Generation failed: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I encountered an error. Please try again.",
                })


def render_sidebar():
    st.sidebar.title("📚 RAG Research Assistant")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload PDF")
    uploaded_file = st.sidebar.file_uploader(
        "Drag and drop a PDF", type=["pdf"],
        help="Upload a research paper to add to your library",
    )
    if uploaded_file is not None:
        if st.sidebar.button("Process Document", type="primary"):
            try:
                file_path = save_uploaded_file(uploaded_file)
                process_pdf(file_path)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Uploaded Documents")
    if st.session_state.ingested_docs:
        for doc in st.session_state.ingested_docs:
            st.sidebar.markdown(f"- ✅ {doc}")
    else:
        st.sidebar.info("No documents uploaded yet")

    if has_chunks():
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Stats")
        st.sidebar.text(f"Total chunks: {count_chunks()}")
        if st.session_state.bm25_index:
            st.sidebar.text("BM25 index: ✅ Ready")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### LLM Providers")
    st.sidebar.caption("Groq is free. Add your own key for other providers:")
    with st.sidebar.expander("Anthropic (optional)"):
        st.text_input("API Key", type="password", key="anthropic_key")
        st.text_input("Model", key="anthropic_model")
    with st.sidebar.expander("OpenAI (optional)"):
        st.text_input("API Key", type="password", key="openai_key")
        st.text_input("Model", key="openai_model")


def render_chat():
    st.markdown("### Related Work Generator")
    st.caption("Describe your research topic and the system will generate a cited Related Work paragraph.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "cited_papers" in message:
                if message["cited_papers"]:
                    st.markdown("---")
                    st.markdown("**Papers Cited:**")
                    for title in message["cited_papers"]:
                        st.markdown(f"- {title}")

    if prompt := st.chat_input("Describe your research topic or paste your abstract..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_query(prompt)


def main():
    init_session_state()
    try:
        Config.validate()
    except EnvironmentError:
        st.sidebar.warning("No LLM API keys found in .env. Add one via the sidebar.")
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()