"""
Article Extractor Service.

Downloads and extracts the full text content from a news article URL
using the newspaper3k library.

Usage:
    from app.services.article_extractor import extract_article
    data = extract_article("https://example.com/article")
"""

import logging
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
