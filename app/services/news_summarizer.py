import json
import logging
import time
from typing import Dict, Any, Optional, List

from app.config import settings
from app.services.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


_CACHE: dict[str, dict] = {}
_CACHE_MAX = 256


def _cache_get(url: str) -> Optional[dict]:
    return _CACHE.get(url)


def _cache_set(url: str, value: dict) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        # naive eviction: remove oldest inserted key
        first_key = next(iter(_CACHE.keys()), None)
        if first_key:
            _CACHE.pop(first_key, None)
    _CACHE[url] = value


SYSTEM_PROMPT = """You are an expert news summarizer.
You will be given ARTICLE text. Your job is to produce a short, neutral summary and key points.

Rules:
- Maximum 2 paragraphs for the summary.
- Do NOT copy large chunks of the article.
- Focus on the main facts: who/what/when/where/why.
- Neutral tone.
- Return ONLY valid JSON in the schema:
{
  "summary": "...",
  "key_points": ["...", "..."]
}
"""


REASONING_PROMPT = """You are an assistant explaining why a news article matters.
Based ONLY on the provided ARTICLE text, explain:
- why this news is important
- what key facts it reports

Rules:
- Max 3 sentences.
- Do NOT copy large chunks.
- Neutral tone.
- Return ONLY JSON:
{ "reasoning": "..." }
"""


def summarize_news_article(article_text: str) -> Dict[str, Any]:
    """
    Summarize an article body into {summary, key_points}.
    """
    text = (article_text or "").strip()
    if not text:
        return {"summary": "", "key_points": []}

    try:
        client = get_openrouter_client()
        resp = client.chat.completions.create(
            model=getattr(settings, "MODEL_NEWS_SUMMARIZER", settings.MODEL_EVIDENCE_ANALYST),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"ARTICLE:\n{text}"},
            ],
            temperature=0.2,
            max_tokens=450,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        summary = str(data.get("summary", "") or "")
        key_points = data.get("key_points", []) or []
        if not isinstance(key_points, list):
            key_points = []
        key_points = [str(k) for k in key_points if k]
        return {"summary": summary, "key_points": key_points}
    except Exception as e:
        logger.warning("News summarization failed; using fallback. error=%s", e)
        # Fallback: naive truncation
        fallback = text[:600].strip()
        return {"summary": fallback, "key_points": []}


def generate_news_reasoning(article_text: str) -> str:
    text = (article_text or "").strip()
    if not text:
        return ""
    try:
        client = get_openrouter_client()
        resp = client.chat.completions.create(
            model=getattr(settings, "MODEL_NEWS_SUMMARIZER", settings.MODEL_EVIDENCE_ANALYST),
            messages=[
                {"role": "system", "content": REASONING_PROMPT},
                {"role": "user", "content": f"ARTICLE:\n{text}"},
            ],
            temperature=0.2,
            max_tokens=220,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        return str(data.get("reasoning", "") or "")
    except Exception as e:
        logger.warning("News reasoning generation failed; using fallback. error=%s", e)
        return ""


def summarize_and_reason(url: str, article_text: str) -> Dict[str, Any]:
    """
    Cached wrapper returning {summary, key_points, reasoning}.
    """
    cached = _cache_get(url)
    if cached:
        return cached

    t0 = time.perf_counter()
    summ = summarize_news_article(article_text)
    reasoning = generate_news_reasoning(article_text)
    out = {**summ, "reasoning": reasoning}
    _cache_set(url, out)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("news_summarizer_cached url=%s elapsed_ms=%s", url, elapsed_ms)
    return out

