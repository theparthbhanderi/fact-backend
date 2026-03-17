import json
import logging
import re
from typing import Dict, Any, List

import requests

from app.services.embedding_service import generate_embedding

logger = logging.getLogger(__name__)


RUMOR_KEYWORDS = ["rumor", "unconfirmed", "viral", "fake", "hoax"]


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _cosine(a, b) -> float:
    import numpy as np

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float((a @ b) / denom)


def _reddit_search(query: str, limit: int = 5) -> Dict[str, Any]:
    url = "https://www.reddit.com/search.json"
    headers = {
        # Public endpoint requires a UA; keep it explicit to avoid 429s.
        "User-Agent": "AIFactChecker/1.0 (SocialSignalAnalyzer)",
        "Accept": "application/json",
    }
    params = {"q": query, "limit": int(limit), "sort": "relevance", "t": "all"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def analyze_social_signals(claim: str) -> Dict[str, Any]:
    """
    Pulls social discussion signals (currently Reddit public search) and computes a rumor score.

    Returns:
    {
      "rumor_detected": bool,
      "discussion_volume": int,
      "misinformation_probability": float,
      "social_sources": [{platform,title,url,snippet}]
    }
    """
    logger.info('social_signal_analysis_started claim=%r', (claim or "")[:120])

    social_sources: List[Dict[str, str]] = []
    discussion_volume = 0

    try:
        data = _reddit_search(claim, limit=5)
        listing = (data or {}).get("data", {}) or {}
        children = listing.get("children", []) or []

        # 'dist' is the total number of results returned by Reddit for this query (approx).
        discussion_volume = int(listing.get("dist") or len(children) or 0)

        for ch in children:
            d = (ch or {}).get("data", {}) or {}
            title = (d.get("title") or "").strip()
            permalink = d.get("permalink") or ""
            url = d.get("url") or ""
            if permalink and permalink.startswith("/"):
                url = "https://www.reddit.com" + permalink
            selftext = (d.get("selftext") or "").strip()
            subreddit = (d.get("subreddit") or "").strip()

            snippet = selftext or title
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if len(snippet) > 300:
                snippet = snippet[:300].rstrip() + "…"

            social_sources.append(
                {
                    "platform": "Reddit",
                    "title": title or "(untitled)",
                    "url": url,
                    "snippet": (f"r/{subreddit}: " if subreddit else "") + snippet,
                }
            )
    except Exception as e:
        logger.warning("Reddit social search failed; continuing without social signals. error=%s", e)

    logger.info("social_posts_found=%s", len(social_sources))

    # Heuristic rumor/misinfo scoring
    text_blob = " ".join([claim] + [s.get("title", "") + " " + s.get("snippet", "") for s in social_sources]).lower()
    rumor_keywords_count = sum(text_blob.count(k) for k in RUMOR_KEYWORDS)
    rumor_kw_norm = min(rumor_keywords_count, 10) / 10.0

    # Normalize volume into [0,1] (cap at 50 to avoid runaway)
    volume_norm = min(max(discussion_volume, 0), 50) / 50.0

    # Uncertainty proxy: question marks and hedging terms in snippets
    uncertainty_hits = 0
    for s in social_sources:
        t = (s.get("title", "") + " " + s.get("snippet", "")).lower()
        if "?" in t or "is it true" in t or "anyone know" in t or "unverified" in t:
            uncertainty_hits += 1
    source_uncertainty_score = (uncertainty_hits / max(1, len(social_sources))) if social_sources else 0.0

    misinfo_probability = _clamp01(
        (rumor_kw_norm * 0.4) + (volume_norm * 0.3) + (source_uncertainty_score * 0.3)
    )

    rumor_detected = bool(discussion_volume > 10 and misinfo_probability > 0.5)

    logger.info(
        'SocialSignals claim=%r posts=%s misinfo_probability=%.2f rumor_detected=%s',
        (claim or "")[:120],
        discussion_volume,
        misinfo_probability,
        rumor_detected,
    )
    logger.info("misinformation_probability=%.4f", misinfo_probability)

    return {
        "rumor_detected": rumor_detected,
        "discussion_volume": discussion_volume,
        "misinformation_probability": float(misinfo_probability),
        "social_sources": social_sources,
    }


def social_sources_to_evidence(claim: str, social_sources: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert social sources into the pipeline evidence object structure.
    Lower credibility than news sources.
    """
    evidence: List[Dict[str, Any]] = []
    claim_vec = generate_embedding(claim)
    for s in social_sources:
        snippet = (s.get("snippet") or "").strip()
        if not snippet:
            continue
        sim = _cosine(generate_embedding(snippet), claim_vec)
        evidence.append(
            {
                "title": s.get("title", ""),
                "url": s.get("url", ""),
                "source": "Reddit",
                "text": snippet,
                "similarity_score": float(_clamp01(sim)),
                "score": float(_clamp01(sim)),
                "credibility_score": 0.4,
                "source_credibility": 0.4,
            }
        )
    return evidence

