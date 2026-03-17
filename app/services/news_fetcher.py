"""
News Fetcher Service.

Searches the internet for news articles related to a given claim
using the NewsAPI (https://newsapi.org).

Usage:
    from app.services.news_fetcher import search_news
    articles = search_news("NASA discovered alien life")
"""

import logging
import requests
from app.config import settings

logger = logging.getLogger(__name__)

# NewsAPI endpoint
NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"


def search_news(query: str, page_size: int = 5) -> list[dict]:
    """
    Search NewsAPI for articles related to the claim.

    Queries the NewsAPI /v2/everything endpoint with the given
    claim text and returns structured article metadata.

    Args:
        query: The news claim or keywords to search for.
        page_size: Maximum number of articles to return (default 5).

    Returns:
        A list of dicts, each containing:
            - title (str): Article headline.
            - url (str): Link to the full article.
            - source (str): Publishing source name.
            - description (str): Short article description.

    Raises:
        RuntimeError: If the NewsAPI request fails.
    """
    logger.info(f"🔍 Searching news for: '{query}'")

    if not settings.NEWS_API_KEY:
        logger.warning("Missing API key; skipping NewsAPI fetch. key=%s", "NEWSAPI_KEY")
        return []

    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": settings.NEWS_API_KEY,
    }

    try:
        response = requests.get(NEWSAPI_EVERYTHING_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            error_msg = data.get("message", "Unknown NewsAPI error")
            logger.error(f"NewsAPI error: {error_msg}")
            raise RuntimeError(f"NewsAPI error: {error_msg}")

        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", ""),
                "description": article.get("description", ""),
            })

        logger.info(f"✅ Found {len(articles)} articles for: '{query}'")
        return articles

    except requests.exceptions.Timeout:
        logger.error("NewsAPI request timed out.")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to NewsAPI.")
        return []
    except requests.exceptions.HTTPError as e:
        logger.error(f"NewsAPI HTTP error: {e}")
        return []
