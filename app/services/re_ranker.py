import logging
from typing import List, Dict
from app.config import settings

logger = logging.getLogger(__name__)

class CrossEncoderReRanker:
    """Uses a lightweight Cross-Encoder to exactly score semantic relevance of chunks to a claim."""
    def __init__(self):
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            # using the model specified in the architecture blueprint
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✅ CrossEncoder model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load CrossEncoder model: {e}")

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Takes a list of chunk dictionaries (must have 'text' key) and scores them against the query.
        Returns the top_k scoring chunks.
        """
        if not chunks or not query:
            return []
            
        if not self.model:
            logger.warning("⚠️ CrossEncoder not loaded. Returning unranked chunks.")
            return chunks[:top_k]

        # Prepare pairs for cross-encoder: (Query, ChunkText)
        pairs = [[query, chunk.get("text", "")] for chunk in chunks]
        
        try:
            scores = self.model.predict(pairs)
            
            raw_scores = [float(s) for s in scores]
            if not raw_scores:
                return chunks[:top_k]

            # Normalize to 0..1 (min-max)
            s_min = min(raw_scores)
            s_max = max(raw_scores)
            denom = (s_max - s_min) if (s_max - s_min) > 1e-9 else 1.0
            norm_scores = [(s - s_min) / denom for s in raw_scores]

            # Attach scores to chunks
            for i, chunk in enumerate(chunks):
                chunk["relevance_score_raw"] = raw_scores[i]
                chunk["relevance_score"] = float(norm_scores[i])
                
            # Sort by score descending
            ranked_chunks = sorted(chunks, key=lambda x: float(x.get("relevance_score", 0.0) or 0.0), reverse=True)

            # Remove very low similarity chunks
            ranked_chunks = [c for c in ranked_chunks if float(c.get("relevance_score", 0.0) or 0.0) >= 0.15]
            return ranked_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Re-ranking failed: {e}")
            return chunks[:top_k]

reranker = CrossEncoderReRanker()

def rerank_evidence(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    return reranker.rerank(query, chunks, top_k)
