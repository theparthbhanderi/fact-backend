"""
Evidence Retriever Service (Persistent RAG).

Orchestrates the retrieval pipeline:
1. Embed the claim.
2. Search persistent FAISS vector database.
3. If sufficient matches exist, return them instantly.
4. Else, fetch live articles via multi-source search, ingest into FAISS, and re-query.
"""

import logging
from app.services.vector_store import VectorStore
from app.services.multi_source_search import multi_source_search

logger = logging.getLogger(__name__)

def retrieve_relevant_evidence(claim: str, top_k: int = 5) -> dict:
    """
    RAG Retrieval Pipeline using persistent FAISS + Live Fallback.
    
    Args:
        claim: The user's claim to fact-check.
        top_k: Number of top documents to retrieve (default 5).
        
    Returns:
        A dict containing: finding 'claim' and 'relevant_articles'.
    """
    logger.info(f"🎯 RAG Retrieval for claim: '{claim}'")
    store = VectorStore()
    
    # 1. Search persistent FAISS
    results = store.search(claim, top_k=top_k)
    
    # 2. Heuristic check: do we have enough *good* context?
    # For a dense embedding match, scores > 0.6 to 0.7 roughly mean strong semantic similarity.
    high_quality_results = [r for r in results if r['score'] > 0.6]
    
    if len(results) >= 3 and len(high_quality_results) >= 1:
        logger.info(f"⚡ FAST PATH RAG: Found {len(results)} matches in local FAISS. Skipping live search.")
        return {"claim": claim, "relevant_articles": results[:top_k]}

    logger.info("⚠️ Insufficient context in persistent DB. Falling back to live search...")
    
    # 3. Live Search
    live_articles = multi_source_search(claim)
    
    if not live_articles:
        logger.warning("Live search returned no articles.")
        return {"claim": claim, "relevant_articles": results} # return whatever we had historically
        
    # 4. Ingest into persistent FAISS
    added = store.add_documents(live_articles)
    if added > 0:
        logger.info(f"📥 Ingested {added} new live articles into persistent FAISS DB.")
    else:
        logger.info("📥 Live search articles were all duplicates, nothing new ingested.")
        
    # 5. Search again to get the semantically ranked top-k from the enhanced DB
    final_results = store.search(claim, top_k=top_k)
    
    return {"claim": claim, "relevant_articles": final_results[:top_k]}
