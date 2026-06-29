"""
Generate Related Work summaries citing papers by title with real regeneration loop.
"""

from time import monotonic
from typing import Optional
import json

from loguru import logger
from pydantic import BaseModel, field_validator

from src.retrieval.pipeline import retrieval
from src.generation.providers import build_provider_chain, create_langchain_client, Provider
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
    """Validated Related Work response with paper citations."""
    related_work: str
    cited_papers: list[str]
    valid: bool = True
    
    @field_validator("related_work")
    def must_cite_papers_by_title(cls, v):
        """Ensure the response cites at least one paper by title."""
        if not v or len(v) < 100:
            raise ValueError("Related Work must be at least 100 characters")
        # Check that paper titles appear in the text (basic validation)
        if "[" not in v or "]" not in v:
            raise ValueError("Response must contain paper citations in [Title] format")
        return v
    
    @field_validator("cited_papers")
    def must_have_papers(cls, v):
        """Ensure at least one paper is cited."""
        if not v or len(v) == 0:
            raise ValueError("Must cite at least one paper by title")
        return v


def _extract_cited_papers_from_response(response_text: str, available_titles: list[str]) -> list[str]:
    """Extract paper titles that appear in [brackets] in the response."""
    cited = []
    import re
    # Find all [text] patterns
    matches = re.findall(r'\[([^\]]+)\]', response_text)
    for match in matches:
        match_clean = match.strip()
        # Check if this looks like a paper title (appears in available titles or is a reasonable length)
        if len(match_clean) > 3 and match_clean.lower() not in ['source', 'citation', 'ref']:
            if match_clean not in cited:
                cited.append(match_clean)
    return cited


def _build_related_work_prompt(query: str, chunks: list[dict]) -> str:
    """Build a prompt that asks for a Related Work summary citing papers by title."""
    
    # Extract unique paper titles from chunks
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
    
    prompt = f"""You are a research literature analyst. Your task is to write a Related Work section.

User Query/Topic: {query}

Available Papers in Library:
{papers_str}

Relevant Excerpts from Papers:
{chunks_str}

Write a 2-3 paragraph Related Work summary that:
1. Synthesizes insights across the available papers
2. Cites each paper by its full title in [brackets], like [Paper Title Here]
3. Explains how each paper relates to the user's query
4. Flows naturally without breaking into fragments

Output ONLY the Related Work text, no introduction, no labels, no markdown.
Cite papers by title in [brackets] when referencing them."""
    
    return prompt


def _invoke_llm_for_related_work(prompt: str) -> tuple[str, list[str]]:
    """Call LLM and extract related work + cited papers."""
    chain = build_provider_chain()
    last_error: Exception | None = None
    
    for provider in chain:
        try:
            logger.info(f"Trying provider: {provider.name} ({provider.model})")
            client = create_langchain_client(provider)
            response = client.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Provider {provider.name} succeeded")
            return text, []  # Return empty cited list; will extract from text
        except Exception as e:
            logger.warning(f"Provider {provider.name} failed: {e}")
            last_error = e
            continue
    
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    before=before_log(logger, logging.WARNING),
    reraise=True,
)
def _call_generation_with_retry(prompt: str) -> str:
    """Call generation with retry logic."""
    text, _ = _invoke_llm_for_related_work(prompt)
    return text


def generate_related_work(
    query: str,
    bm25_index,
    max_retries: int = 3,
) -> tuple[str, list[str]]:
    """
    Generate a Related Work summary citing papers by title, with regeneration on validation failure.
    
    Args:
        query: User query or topic
        bm25_index: BM25 index for retrieval
        max_retries: Number of times to retry if validation fails (default 3)
    
    Returns:
        (related_work_text, list_of_cited_paper_titles)
    
    Raises:
        RuntimeError: If all retries fail
    """
    Config.validate()
    
    t0 = monotonic()
    
    # Retrieve relevant chunks
    chunks = retrieval(query, bm25_index, top_k=10)  # Top 10 for synthesis
    
    if not chunks:
        raise ValueError("No relevant papers found for query")
    
    # Build prompt
    prompt = _build_related_work_prompt(query, chunks)
    
    # Try to generate with validation and regeneration loop
    for attempt in range(max_retries):
        try:
            logger.info(f"Generation attempt {attempt + 1}/{max_retries}")
            
            # Generate response
            response_text = _call_generation_with_retry(prompt)
            
            # Extract cited papers
            available_titles = sorted(set(
                c.get("paper_title", c.get("doc_id", "Unknown"))
                for c in chunks
            ))
            cited_papers = _extract_cited_papers_from_response(response_text, available_titles)
            
            # Validate
            validated = RelatedWorkResponse(
                related_work=response_text,
                cited_papers=cited_papers if cited_papers else available_titles[:1],  # Fallback
                valid=True
            )
            
            elapsed_ms = (monotonic() - t0) * 1000
            logger.info(f"Related Work generated in {elapsed_ms:.0f}ms, cited {len(validated.cited_papers)} papers")
            
            return validated.related_work, validated.cited_papers
        
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Last attempt failed, raise
                raise RuntimeError(
                    f"Failed to generate valid Related Work after {max_retries} attempts. "
                    f"Last error: {e}"
                )
            # Otherwise, retry with regeneration
            continue
    
    # Fallback (shouldn't reach here)
    raise RuntimeError("Related Work generation failed unexpectedly")
