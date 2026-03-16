"""
Evidence Collector Service.

Orchestrates the news search and article extraction pipeline to
gather structured evidence for a given claim. This module combines
the news_fetcher and article_extractor services.

Pipeline:
    1. Search NewsAPI for relevant articles.
    2. Extract full text from each article URL.
    3. Return combined structured evidence.

Usage:
    from app.services.evidence_collector import collect_evidence
    result = collect_evidence("NASA discovered alien life")
"""

import logging
from app.services.news_fetcher import search_news
from app.services.article_extractor import extract_article

logger = logging.getLogger(__name__)

# Maximum number of articles to extract
MAX_ARTICLES = 5


def collect_evidence(claim: str) -> dict:
    """
    Collect evidence articles for a given claim.

    Searches for news articles related to the claim, extracts
    full text content from each, and returns a combined result.

    Args:
        claim: The news claim to gather evidence for.

    Returns:
        A dict containing:
            - claim (str): The original claim.
            - articles (list[dict]): Extracted article data, each with:
                - title (str): Article headline.
                - url (str): Article URL.
                - text (str): Full article text.
                - source (str): Publisher name.
                - summary (str): Auto-generated summary.
                - publish_date (str): Publication date.

    Note:
        Articles that fail extraction are filtered out. The pipeline
        continues even if some articles cannot be extracted.
    """
    logger.info(f"📦 Collecting evidence for claim: '{claim}'")

    # Step 1: Search for news articles
    try:
        raw_articles = search_news(claim, page_size=MAX_ARTICLES)
    except RuntimeError as e:
        logger.error(f"News search failed: {e}")
        return {"claim": claim, "articles": []}

    if not raw_articles:
        logger.warning("No articles found for this claim.")
        return {"claim": claim, "articles": []}

    # Step 2: Extract content from each article
    extracted_articles = []
    for article_meta in raw_articles:
        url = article_meta.get("url", "")
        if not url:
            continue

        extracted = extract_article(url)

        # Only include articles that have meaningful text
        if extracted.get("text") and len(extracted["text"]) > 50:
            extracted_articles.append({
                "title": extracted.get("title") or article_meta.get("title", ""),
                "url": url,
                "text": extracted["text"],
                "source": article_meta.get("source", ""),
                "summary": extracted.get("summary", ""),
                "publish_date": extracted.get("publish_date", ""),
            })
        else:
            logger.warning(
                f"Skipped article (insufficient text): {url}"
            )

    logger.info(
        f"✅ Collected {len(extracted_articles)} articles "
        f"out of {len(raw_articles)} found."
    )

    return {
        "claim": claim,
        "articles": extracted_articles,
    }
