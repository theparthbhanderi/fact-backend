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
from bs4 import BeautifulSoup

from app.services.http_client import http_client
from app.services.cache_service import get_or_set_json

logger = logging.getLogger(__name__)

_BOILERPLATE_TAGS = {"nav", "aside", "footer", "header", "form"}
_BOILERPLATE_CLASS_ID_HINTS = (
    "cookie",
    "consent",
    "subscribe",
    "newsletter",
    "promo",
    "advert",
    "ads",
    "sidebar",
    "menu",
    "navbar",
    "header",
    "footer",
    "modal",
    "popup",
    "paywall",
)


def _readability_clean_html(html: str) -> str:
    """
    Readability-ish fallback: drop boilerplate and keep article-like text.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    for t in soup(["script", "style", "noscript", "svg"]):
        t.decompose()

    # Remove boilerplate tags
    for tag in list(soup.find_all(_BOILERPLATE_TAGS)):
        tag.decompose()

    # Remove obvious boilerplate blocks by id/class hints
    for el in list(soup.find_all(True)):
        ident = " ".join(
            [
                (el.get("id") or ""),
                " ".join(el.get("class") or []),
            ]
        ).lower()
        if ident and any(h in ident for h in _BOILERPLATE_CLASS_ID_HINTS):
            try:
                el.decompose()
            except Exception:
                pass

    # Prefer <article>
    root = soup.find("article") or soup.body or soup
    if not root:
        return ""

    parts = []
    for p in root.find_all(["p", "h1", "h2", "h3", "li"]):
        txt = " ".join((p.get_text(" ", strip=True) or "").split())
        if len(txt) < 40:
            continue
        parts.append(txt)

    text = "\n\n".join(parts)
    # final cleanup
    text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
    return text.strip()


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

    if not url or not url.strip():
        return {"url": url or "", "title": "", "text": "", "summary": "", "publish_date": ""}

    def _build():
        return _extract_article_uncached(url.strip())

    # Cache full extraction for 24h
    return get_or_set_json("article_extract", {"url": url.strip()}, _build, ttl_s=24 * 3600)


def _extract_article_uncached(url: str) -> dict:
    """
    Uncached extraction (called via disk cache wrapper).
    """
    # Always return all fields.
    base = {"url": url, "title": "", "text": "", "summary": "", "publish_date": ""}

    try:
        # 1. Try Jina Reader API (Best for clean Markdown extraction)
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "application/json",
            "X-Return-Format": "markdown"
        }
        
        data = http_client.get_json(jina_url, headers=headers)
        if isinstance(data, dict):
            jina_title = data.get("data", {}).get("title", "")
            jina_text = data.get("data", {}).get("content", "")
            if jina_text and len(jina_text.strip()) > 50:
                logger.info(f"✅ Extracted {len(jina_text)} chars via Jina API: {url}")
                base["title"] = jina_title or ""
                base["text"] = jina_text or ""
                base["summary"] = data.get("data", {}).get("description", "") or ""
                return base
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

        # If newspaper got some text, keep it; else fall through to BS4 fallback
        if extracted["text"] and len(extracted["text"].strip()) > 80:
            return extracted

    except Exception as e:
        logger.warning(f"⚠️ newspaper3k failed for {url}: {e}. Falling back to BeautifulSoup.")

    # 3. BeautifulSoup readability-style fallback
    try:
        html = http_client.get_text(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AIFactChecker/1.0)"})
        cleaned = _readability_clean_html(html)
        if cleaned and len(cleaned.strip()) > 80:
            title = ""
            try:
                soup = BeautifulSoup(html, "html.parser")
                title = (soup.title.get_text(strip=True) if soup.title else "") or ""
            except Exception:
                title = ""
            base["title"] = title
            base["text"] = cleaned
            return base
    except Exception as e:
        logger.error(f"❌ BeautifulSoup fallback failed for {url}: {e}")

    return base
