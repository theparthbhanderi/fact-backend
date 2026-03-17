"""
Evidence Retriever Service (Persistent RAG with Cross-Encoder Re-Ranking).

Orchestrates the retrieval pipeline:
1. Embed the claim.
2. Search persistent FAISS vector database.
3. If sufficient matches exist, return them instantly.
4. Else, fetch live articles via multi-source search.
5. Download and extract articles in parallel (Jina Reader).
6. Chunk text and ingest into FAISS.
7. Re-query to get top 15.
8. Re-rank top 15 using Cross-Encoder to find top 5.
"""

import logging
import concurrent.futures
from typing import Dict, List, Any
import urllib.parse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from app.services.vector_store import VectorStore
from app.services.multi_source_search import multi_source_search
from app.services.article_extractor import extract_article
from app.services.chunk_service import chunk_text
from app.services.re_ranker import rerank_evidence
from app.services.evidence_extractor import extract_relevant_sentences, dedupe_by_sentence_similarity
from app.services.source_credibility import get_source_credibility

logger = logging.getLogger(__name__)

def _dedupe_urls(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for it in items or []:
        url = (it.get("url") or "").strip()
        key = url.lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _top_articles_from_sentences(sentences: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    """
    Build a compact article list from sentence evidence.
    """
    def _domain(url: str) -> str:
        try:
            netloc = urllib.parse.urlparse(url).netloc.lower()
            return netloc[4:] if netloc.startswith("www.") else netloc
        except Exception:
            return ""

    def _recency_score(published_at: str) -> float:
        """
        0..1 where 1 = within ~7 days, 0 = very old/unknown.
        """
        s = (published_at or "").strip()
        if not s:
            return 0.0
        dt = None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            try:
                dt = parsedate_to_datetime(s)
            except Exception:
                dt = None
        if not dt:
            return 0.0
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
        if age_days <= 7:
            return 1.0
        if age_days >= 90:
            return 0.0
        return max(0.0, 1.0 - ((age_days - 7.0) / (90.0 - 7.0)))

    by_url: Dict[str, Dict[str, Any]] = {}
    for e in sentences or []:
        url = (e.get("url") or "").strip()
        if not url:
            continue
        cur = by_url.get(url)
        score = float(e.get("similarity_score", e.get("score", 0.0)) or 0.0)
        if not cur or score > float(cur.get("similarity_score", 0.0) or 0.0):
            pub = (e.get("published_at") or "").strip()
            cred = float(e.get("source_credibility", 0.0) or 0.0)
            rec = _recency_score(pub)
            # Composite for ordering: credibility + recency + similarity, with a small diversity boost later.
            composite = 0.45 * cred + 0.35 * score + 0.20 * rec
            by_url[url] = {
                "title": e.get("title", ""),
                "source": e.get("source", ""),
                "url": url,
                "snippet": e.get("text", ""),
                "similarity_score": score,
                "source_credibility": cred,
                "published_at": pub,
                "recency_score": rec,
                "domain": _domain(url),
                "composite_score": composite,
            }
    arts = list(by_url.values())
    arts.sort(key=lambda x: float(x.get("composite_score", 0.0) or 0.0), reverse=True)

    # Domain diversity boost: pick at most 2 per domain in final 8
    out: List[Dict[str, Any]] = []
    per_domain: Dict[str, int] = {}
    for a in arts:
        dom = a.get("domain") or ""
        if dom:
            if per_domain.get(dom, 0) >= 2:
                continue
            per_domain[dom] = per_domain.get(dom, 0) + 1
        out.append(a)
        if len(out) >= limit:
            break
    return out


def retrieve_relevant_evidence(claim: str, queries: List[str], top_k: int = 5) -> Dict[str, Any]:
    """
    RAG Retrieval Pipeline using FAISS, Parallel Article Extraction, and Cross-Encoder Re-Ranking.
    """
    logger.info(f"🎯 RAG Retrieval for claim: '{claim}'")
    store = VectorStore()
    
    # 1. Start with historically stored FAISS chunks
    historical_results = store.search(claim, top_k=15)
    
    high_quality_results = [r for r in historical_results if r.get('score', 0) > 0.6]
    
    if len(historical_results) >= 5 and len(high_quality_results) >= 3:
        logger.info(f"⚡ FAST PATH RAG: Found {len(historical_results)} good semantic matches in FAISS. Re-ranking...")
        ranked_top_k = rerank_evidence(claim, historical_results, top_k=top_k)
        sentence_evidence = _build_sentence_evidence_from_items(claim, ranked_top_k)
        return {
            "claim": claim,
            "top_articles": _top_articles_from_sentences(sentence_evidence, limit=8),
            "top_sentences": sentence_evidence[:12],
            # backward compatibility
            "relevant_articles": sentence_evidence,
        }

    logger.info("⚠️ Insufficient context in persistent DB. Falling back to live search...")
    
    # 2. Live Search
    live_articles = multi_source_search(queries)
    
    if not live_articles:
        logger.warning("Live search returned no articles.")
        # Fallback to whatever we had
        ranked_fail = rerank_evidence(claim, historical_results, top_k=top_k) if historical_results else []
        sentence_evidence = _build_sentence_evidence_from_items(claim, ranked_fail) if ranked_fail else []
        return {
            "claim": claim,
            "top_articles": _top_articles_from_sentences(sentence_evidence, limit=8),
            "top_sentences": sentence_evidence[:12],
            "relevant_articles": sentence_evidence,
        }
        
    # 3. Parallel Article Extraction (Jina Reader)
    logger.info(f"📥 Extracting {len(live_articles)} articles in parallel...")
    extracted_data = []
    
    def fetch_and_parse(article_meta: Dict) -> Dict:
        url = article_meta.get("url")
        if not url:
            return None
        full_art = extract_article(url)
        if full_art and full_art.get("text"):
            # Merge API metadata with extracted text
            merged = article_meta.copy()
            merged["text"] = full_art["text"]
            merged["summary"] = full_art.get("summary", "")
            return merged
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_and_parse, art) for art in live_articles]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                extracted_data.append(res)
                
    # 4. Chunk Extracted Markdown Documents
    all_chunks = []
    for art in extracted_data:
        chunks = chunk_text(art["text"], art, chunk_size=800, chunk_overlap=100)
        all_chunks.extend(chunks)
        
    # 5. Ingest into Persistent FAISS database
    if all_chunks:
        added = store.add_documents(all_chunks)
        logger.info(f"📥 Ingested {added} new article chunks into FAISS.")
    else:
        logger.info("📥 No valid chunks extracted from live articles.")
        
    # 6. Re-query FAISS for the broadest top 15 matches (now featuring the new data)
    enhanced_results = store.search(claim, top_k=15)
    
    # 7. Cross-Encoder Re-Ranking to find absolute best 5 chunks
    if not enhanced_results:
        return {"claim": claim, "top_articles": [], "top_sentences": [], "relevant_articles": []}
        
    logger.info(f"⚖️ Re-ranking Top 15 semantic matches with Cross-Encoder...")
    best_chunks = rerank_evidence(claim, enhanced_results, top_k=top_k)

    sentence_evidence = _build_sentence_evidence_from_items(claim, best_chunks)
    return {
        "claim": claim,
        "top_articles": _top_articles_from_sentences(sentence_evidence, limit=8),
        "top_sentences": sentence_evidence[:12],
        "relevant_articles": sentence_evidence,
    }


def _cap_sentence(s: str, max_len: int = 300) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len].rstrip() + "…"


def _build_sentence_evidence_from_items(claim: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert chunk/article items (with text) into claim-focused sentence evidence objects.
    - Extract top 5 sentences per item.
    - Deduplicate globally by sentence-to-sentence sim > 0.95.
    - Keep top 12 sentences globally by similarity to claim.
    - Add quality metrics: similarity_score, source_credibility, evidence_rank.
    """
    if not items:
        return []

    extracted_sentences: List[Dict[str, Any]] = []

    for it in items:
        base_text = it.get("text") or it.get("content") or ""
        if not base_text:
            continue
        title = it.get("title", "")
        url = it.get("url", "")
        source = it.get("source", "Unknown")
        published_at = it.get("published_at") or it.get("publish_date") or ""

        per_item = extract_relevant_sentences(claim, base_text, top_n=5)
        for s in per_item:
            sentence = _cap_sentence(s.get("sentence", ""))
            sim = float(s.get("similarity", 0.0) or 0.0)
            extracted_sentences.append(
                {
                    "title": title,
                    "url": url,
                    "source": source,
                    "published_at": published_at,
                    "text": sentence,
                    "score": sim,  # keeps compatibility with existing downstream code
                    "similarity_score": sim,
                    "source_credibility": get_source_credibility(source, url),
                }
            )

    logger.info(
        "EvidenceExtraction claim=%r sentences_scanned=%s",
        claim[:120],
        len(extracted_sentences),
    )

    # Sort by similarity to claim before dedup
    extracted_sentences.sort(key=lambda x: float(x.get("similarity_score", 0.0) or 0.0), reverse=True)

    # Deduplicate by sentence-to-sentence similarity
    dedupe_payload = [{"sentence": e.get("text", ""), "similarity": e.get("similarity_score", 0.0)} for e in extracted_sentences]
    deduped_payload = dedupe_by_sentence_similarity(dedupe_payload, threshold=0.95)

    # Keep only one evidence object per kept sentence (first occurrence wins)
    kept_sentences = [d.get("sentence", "") for d in deduped_payload if d.get("sentence")]
    first_by_sentence: Dict[str, Dict[str, Any]] = {}
    for e in extracted_sentences:
        txt = e.get("text", "")
        if not txt or txt not in kept_sentences:
            continue
        if txt not in first_by_sentence:
            first_by_sentence[txt] = e
    deduped = [first_by_sentence[s] for s in kept_sentences if s in first_by_sentence]

    logger.info(
        "EvidenceExtractionDedupSummary claim=%r before=%s after=%s",
        claim[:120],
        len(extracted_sentences),
        len(deduped),
    )

    # Global top 12
    deduped.sort(key=lambda x: float(x.get("similarity_score", 0.0) or 0.0), reverse=True)
    final = deduped[:12]

    for idx, e in enumerate(final, 1):
        e["evidence_rank"] = idx

    logger.info(
        "EvidenceExtractionFinal claim=%r top_selected=%s deduplicated_sentences=%s",
        claim[:120],
        len(final),
        len(deduped),
    )
    return final
