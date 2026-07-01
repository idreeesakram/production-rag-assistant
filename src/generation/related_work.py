"""
Generate Related Work summaries citing papers by title with real regeneration loop.
"""

from time import monotonic
from typing import Optional
import re
import difflib

from loguru import logger
from pydantic import BaseModel, field_validator

from src.retrieval.pipeline import retrieval
from src.generation.providers import build_provider_chain, create_langchain_client
from src.config import Config
import logging
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_log,
)


class RelatedWorkResponse(BaseModel):
    related_work: str
    cited_papers: list[str]
    valid: bool = True

    @field_validator("related_work")
    def must_cite_papers_by_title(cls, v):
        if not v or len(v) < 100:
            raise ValueError("Related Work must be at least 100 characters")
        if "[" not in v or "]" not in v:
            raise ValueError("Response must contain paper citations in [Title] format")
        return v

    @field_validator("cited_papers")
    def must_have_papers(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Must cite at least one paper by title")
        return v


def _find_closest_title(match: str, available_titles: list[str], threshold: float = 0.8) -> str | None:
    """Fuzzy match a bracketed string against available paper titles."""
    if not available_titles:
        return None
    matches = difflib.get_close_matches(match, available_titles, n=1, cutoff=threshold)
    return matches[0] if matches else None


def _extract_cited_papers_from_response(response_text: str, available_titles: list[str]) -> list[str]:
    """
    Extract paper titles from [brackets] that actually match available_titles.
    Uses fuzzy matching to handle minor typos.
    """
    cited = []
    matches = re.findall(r'\[([^\]]+)\]', response_text)
    for match in matches:
        match_clean = match.strip()
        if len(match_clean) <= 3:
            continue
        if match_clean.lower() in ['source', 'citation', 'ref']:
            continue
        best_match = _find_closest_title(match_clean, available_titles, threshold=0.8)
        if best_match and best_match not in cited:
            cited.append(best_match)
    return cited


def _build_related_work_prompt(query: str, chunks: list[dict], error_feedback: str = "") -> str:
    """Build prompt for Related Work summary. Optionally includes error feedback on retry."""
    paper_titles = sorted(set(
        c.get("paper_title", c.get("doc_id", "Unknown Paper"))
        for c in chunks
    ))
    papers_str = "\n".join(f"- {title}" for title in paper_titles)
    chunks_str = "\n\n".join([
        f"Paper: {c.get('paper_title', c.get('doc_id', 'Unknown'))}\n"
        f"Page {c.get('page_num', 'N/A')}: {c.get('text', '')[:300]}..."
        for c in chunks
    ])

    feedback_section = ""
    if error_feedback:
        feedback_section = f"\n\nPREVIOUS ATTEMPT FAILED: {error_feedback}\nYou MUST cite papers using their exact titles from the list above in [brackets].\n"

    prompt = f"""You are a research literature analyst. Your task is to write a Related Work section.

User Query/Topic: {query}

Available Papers in Library (cite these by their EXACT titles in [brackets]):
{papers_str}
{feedback_section}
Relevant Excerpts from Papers:
{chunks_str}

Write a 2-3 paragraph Related Work summary that:
1. Synthesizes insights across the available papers
2. Cites each paper by its EXACT full title in [brackets], like [Attention Is All You Need]
3. Only cite papers from the Available Papers list above — do not invent titles
4. Explains how each paper relates to the user's query
5. Flows naturally without breaking into fragments

Output ONLY the Related Work text, no introduction, no labels, no markdown."""

    return prompt


def _invoke_llm_for_related_work(prompt: str) -> str:
    """Call LLM providers in order, return response text."""
    chain = build_provider_chain()
    last_error: Exception | None = None
    for provider in chain:
        try:
            logger.info(f"Trying provider: {provider.name} ({provider.model})")
            client = create_langchain_client(provider)
            response = client.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Provider {provider.name} succeeded")
            return text
        except Exception as e:
            logger.warning(f"Provider {provider.name} failed: {e}")
            last_error = e
            continue
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def generate_related_work(
    query: str,
    bm25_index,
    max_retries: int = 2,
) -> tuple[str, list[str]]:
    """
    Generate a Related Work summary citing papers by title.
    On validation failure, retries with error feedback in the prompt.

    Args:
        query: User query or topic
        bm25_index: BM25 index for retrieval
        max_retries: Number of retry attempts on validation failure (default 2)

    Returns:
        (related_work_text, list_of_cited_paper_titles)

    Raises:
        RuntimeError: If all retries fail
    """
    Config.validate()
    t0 = monotonic()

    chunks = retrieval(query, bm25_index, top_k=10)
    if not chunks:
        raise ValueError("No relevant papers found for query")

    available_titles = sorted(set(
        c.get("paper_title", c.get("doc_id", "Unknown"))
        for c in chunks
    ))

    error_feedback = ""
    for attempt in range(max_retries):
        try:
            logger.info(f"Generation attempt {attempt + 1}/{max_retries}")
            prompt = _build_related_work_prompt(query, chunks, error_feedback=error_feedback)
            response_text = _invoke_llm_for_related_work(prompt)
            cited_papers = _extract_cited_papers_from_response(response_text, available_titles)

            if not cited_papers:
                raise ValueError(
                    f"No valid paper citations found. The response must cite papers "
                    f"from: {available_titles} using [brackets]."
                )

            validated = RelatedWorkResponse(
                related_work=response_text,
                cited_papers=cited_papers,
                valid=True,
            )

            elapsed_ms = (monotonic() - t0) * 1000
            logger.info(f"Related Work generated in {elapsed_ms:.0f}ms, cited {len(validated.cited_papers)} papers")
            return validated.related_work, validated.cited_papers

        except Exception as e:
            error_feedback = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Failed to generate valid Related Work after {max_retries} attempts. "
                    f"Last error: {e}"
                )

    raise RuntimeError("Related Work generation failed unexpectedly")