"""
URL Scraper Service.

Fetches the main content of a news article given its URL.
Uses the free Jina Reader API (https://r.jina.ai/) to extract clean Markdown text.
Falls back to newspaper3k if the API fails.
"""

import logging
import requests
from newspaper import Article

logger = logging.getLogger(__name__)

def fetch_article_content(url: str) -> dict:
    """
    Fetch and clean article content from a URL.
    
    Args:
        url: The web address of the news article.
        
    Returns:
        dict containing 'title', 'text', and 'source'.
    """
    logger.info(f"🌐 Fetching article content from: {url}")
    
    # 1. Try Jina Reader API (Excellent for clean text extraction)
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "application/json",
            "X-Return-Format": "markdown"
        }
        
        response = requests.get(jina_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            title = data.get("data", {}).get("title", "")
            text = data.get("data", {}).get("content", "")
            
            if text and len(text.strip()) > 50:
                logger.info(f"✅ Successfully fetched article via Jina API (Title: {title})")
                return {
                    "title": title or "Unknown Title",
                    "text": text.strip(),
                    "source": url
                }
    except Exception as e:
        logger.warning(f"⚠️ Jina API fetch failed: {e}. Falling back to newspaper3k.")
        
    # 2. Fallback to newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        text = article.text
        title = article.title
        
        if text and len(text.strip()) > 50:
            logger.info(f"✅ Successfully fetched article via newspaper3k (Title: {title})")
            return {
                "title": title or "Unknown Title",
                "text": text.strip(),
                "source": url
            }
        else:
            raise ValueError("Extracted text is empty or too short.")
            
    except Exception as e:
        logger.error(f"❌ Failed to extract article content: {e}")
        raise RuntimeError(f"Could not extract article content from {url}. The site might be blocking access.")
