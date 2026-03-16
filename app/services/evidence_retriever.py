"""
Evidence Retriever Service.

Orchestrates the embedding + vector search pipeline:

    1. Takes the claim and extracted articles from STEP 1.
    2. Embeds all article texts into vectors.
    3. Stores them in a FAISS index.
    4. Embeds the claim and performs similarity search.
    5. Returns the top-k most relevant articles with scores.

This module bridges STEP 1 (evidence collection) and STEP 3
(RAG + LLM analysis) by providing semantically ranked evidence.

Usage:
    from app.services.evidence_retriever import retrieve_relevant_evidence

    result = retrieve_relevant_evidence(claim, articles)
"""

import logging
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def retrieve_relevant_evidence(
    claim: str,
    articles: list[dict],
    top_k: int = 3,
) -> dict:
    """
    Retrieve the most relevant evidence articles for a claim.

    Creates a FAISS index from the provided articles, embeds
    the claim, and performs semantic similarity search to find
    the top-k most relevant evidence pieces.

    Args:
        claim: The user's claim to fact-check.
        articles: List of article dicts from the evidence collector,
                  each containing 'text', 'title', 'url', 'source'.
        top_k: Number of top results to return (default 3).

    Returns:
        A dict containing:
            - claim (str): The original claim.
            - relevant_articles (list[dict]): Top-k articles ranked
              by semantic similarity, each with a 'score' field.
    """
    logger.info(
        f"🎯 Retrieving relevant evidence for: '{claim}' "
        f"from {len(articles)} articles"
    )

    if not articles:
        logger.warning("No articles provided for retrieval.")
        return {"claim": claim, "relevant_articles": []}

    # Step 1-2: Create vector store and add article embeddings
    store = VectorStore()
    added = store.add_documents(articles)

    if added == 0:
        logger.warning("No valid articles could be embedded.")
        return {"claim": claim, "relevant_articles": []}

    # Step 3-4: Embed the claim and search for similar articles
    results = store.search(claim, top_k=top_k)

    logger.info(
        f"✅ Retrieved {len(results)} relevant articles "
        f"(top score: {results[0]['score'] if results else 'N/A'})"
    )

    return {
        "claim": claim,
        "relevant_articles": results,
    }
