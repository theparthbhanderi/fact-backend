"""
Article Extractor Service.

Downloads and extracts the full text content from a news article URL
using the newspaper3k library.

Usage:
    from app.services.article_extractor import extract_article
    data = extract_article("https://example.com/article")
"""

import logging
import requests
from newspaper import Article

logger = logging.getLogger(__name__)


def extract_article(url: str) -> dict:
    """
    Extract the full article content from a news URL.

    Downloads the web page, parses the HTML, and extracts the
    main article text, title, summary, and publish date using
    the newspaper3k library.

    Args:
        url: The full URL of the article to extract.

    Returns:
        A dict containing:
            - url (str): The original article URL.
            - title (str): Extracted headline.
            - text (str): Full article body text.
            - summary (str): Auto-generated summary.
            - publish_date (str): Publication date (ISO format or "").

    Note:
        If extraction fails, returns a dict with empty text and
        an error flag so the pipeline can continue with other articles.
    """
    logger.info(f"📄 Extracting article: {url}")

    try:
        # 1. Try Jina Reader API (Best for clean Markdown extraction)
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "application/json",
            "X-Return-Format": "markdown"
        }
        
        response = requests.get(jina_url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            jina_title = data.get("data", {}).get("title", "")
            jina_text = data.get("data", {}).get("content", "")
            if jina_text and len(jina_text.strip()) > 50:
                logger.info(f"✅ Extracted {len(jina_text)} chars via Jina API: {url}")
                return {
                    "url": url,
                    "title": jina_title,
                    "text": jina_text,
                    "summary": data.get("data", {}).get("description", ""),
                    "publish_date": "",
                }
    except Exception as e:
        logger.warning(f"⚠️ Jina API fetch failed: {e}. Falling back to newspaper3k.")

    try:
        # 2. Fallback to newspaper3k
        article = Article(url)
        article.download()
        article.parse()

        # Run NLP for summary generation
        try:
            article.nlp()
            summary = article.summary
        except Exception:
            summary = ""

        # Format publish date
        pub_date = ""
        if article.publish_date:
            pub_date = article.publish_date.isoformat()

        extracted = {
            "url": url,
            "title": article.title or "",
            "text": article.text or "",
            "summary": summary,
            "publish_date": pub_date,
        }

        if extracted["text"]:
            logger.info(
                f"✅ Extracted {len(extracted['text'])} chars from: {url}"
            )
        else:
            logger.warning(f"⚠️ No text extracted from: {url}")

        return extracted

    except Exception as e:
        logger.error(f"❌ Failed to extract article from {url}: {e}")
        return {
            "url": url,
            "title": "",
            "text": "",
            "summary": "",
            "publish_date": "",
        }
