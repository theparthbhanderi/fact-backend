import logging
import html
import re
import requests
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.multi_source_search import search_google_news, search_gnews, search_newsdata, search_newsapi
from app.services.article_extractor import extract_article
from app.services.news_summarizer import summarize_and_reason

logger = logging.getLogger(__name__)

router = APIRouter(tags=["News"])

_TAG_RE = re.compile(r"<[^>]+>")


def _resolve_final_url(url: str) -> str:
    """
    Resolve Google News / RSS redirect links to the real publisher URL.
    This improves extraction success for `news.google.com/rss/articles/...` links.
    """
    u = (url or "").strip()
    if not u:
        return u

    # Only resolve known redirectors (keeps normal URLs fast)
    if "news.google.com/" not in u:
        return u

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }
    try:
        # HEAD is faster but not always supported; fallback to GET.
        resp = requests.head(u, allow_redirects=True, timeout=10, headers=headers)
        final = (resp.url or "").strip()
        return final or u
    except Exception:
        try:
            resp = requests.get(u, allow_redirects=True, timeout=10, headers=headers)
            final = (resp.url or "").strip()
            return final or u
        except Exception:
            return u


def _clean_snippet(text: str, max_len: int = 220) -> str:
    """
    Google News RSS often returns HTML in `description`.
    Clean it into a readable short snippet for UI cards.
    """
    if not text:
        return ""
    s = html.unescape(text)
    s = _TAG_RE.sub(" ", s)
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


@router.get("/news/search")
async def news_search(q: str = Query(..., min_length=1, max_length=120)) -> Dict[str, Any]:
    """
    Search live news sources and return top 4 results.
    """
    topic = q.strip()
    try:
        results: List[Dict[str, Any]] = []
        # Prefer sources that return direct publisher URLs (best for /news/read)
        results.extend(search_newsapi(topic, limit=4) or [])
        results.extend(search_gnews(topic, limit=4) or [])
        results.extend(search_newsdata(topic, limit=4) or [])
        results.extend(search_google_news(topic, limit=4) or [])

        # De-dup by URL and normalize fields
        seen = set()
        out = []
        for r in results:
            url = r.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(
                {
                    "title": r.get("title", "") or "",
                    "url": url,
                    "source": r.get("source", "") or "",
                    "published_at": r.get("published_at", "") or "",
                    "image": r.get("image", "") or "",
                    "description": _clean_snippet(r.get("description", "") or ""),
                }
            )
            if len(out) >= 4:
                break

        return {"articles": out}
    except Exception as e:
        logger.error("News search failed: %s", e)
        raise HTTPException(status_code=500, detail="News search failed")


@router.get("/news/read")
async def news_read(
    url: str = Query(..., min_length=8, max_length=2048),
    title: Optional[str] = Query(None, max_length=300),
    source: Optional[str] = Query(None, max_length=120),
    published_at: Optional[str] = Query(None, max_length=120),
    description: Optional[str] = Query(None, max_length=1200),
) -> Dict[str, Any]:
    """
    Read an article URL, extract content, summarize + reasoning.
    Never returns full article text.
    """
    requested_url = url.strip()
    source_url = _resolve_final_url(requested_url)
    extracted = extract_article(source_url)
    extracted_title = extracted.get("title", "") or ""
    extracted_published = extracted.get("publish_date", "") or ""
    text = (extracted.get("text", "") or "").strip()

    # If we cannot extract full text (common for Google News redirect URLs),
    # fall back to the short snippet passed from /news/search (still never show full article).
    fallback_text = (description or "").strip()
    content_for_summary = (text or fallback_text).strip()

    if not content_for_summary:
        raise HTTPException(status_code=422, detail="Could not extract article text")

    limited_text = content_for_summary[:3000]
    sr = summarize_and_reason(source_url, limited_text)

    chosen_title = (extracted_title or "").strip()
    if chosen_title.lower() in ("google news", "news", "article"):
        chosen_title = ""

    return {
        "title": (chosen_title or (title or "")).strip(),
        "source": (source or "").strip(),
        "summary": sr.get("summary", "") or "",
        "reasoning": sr.get("reasoning", "") or "",
        "source_url": source_url,
        "published_at": (extracted_published or (published_at or "")).strip(),
        "image": "",
    }

