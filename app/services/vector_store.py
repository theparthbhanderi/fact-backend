"""
Vector Store Service (Lightweight — no FAISS).

Uses scikit-learn cosine_similarity instead of FAISS for
semantic search. Works with TF-IDF embeddings from
embedding_service.py.

Memory usage: ~50MB vs 3GB+ with FAISS + PyTorch.
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.services.embedding_service import (
    create_vectorizer,
    generate_embedding,
    generate_embeddings,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Lightweight vector store using TF-IDF + cosine similarity.

    Replaces FAISS to avoid PyTorch/sentence-transformers memory overhead.
    """

    def __init__(self):
        """Initialize an empty vector store."""
        self.embeddings: np.ndarray | None = None
        self.metadata: list[dict] = []
        logger.info("📦 Initialized lightweight VectorStore (TF-IDF + cosine)")

    def add_documents(self, documents: list[dict]) -> int:
        """
        Embed and add documents to the store.

        Each document dict should have: text, title, url, source.

        Args:
            documents: List of document dicts.

        Returns:
            Number of documents successfully added.
        """
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return 0

        valid_docs = [d for d in documents if d.get("text", "").strip()]
        if not valid_docs:
            logger.warning("All documents have empty text, nothing to add.")
            return 0

        # Extract texts and fit TF-IDF on this corpus
        texts = [d["text"] for d in valid_docs]
        create_vectorizer(texts)

        # Generate embeddings
        self.embeddings = generate_embeddings(texts)

        # Store metadata
        for doc in valid_docs:
            self.metadata.append({
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "text": doc.get("text", ""),
                "source": doc.get("source", ""),
                "summary": doc.get("summary", ""),
                "publish_date": doc.get("publish_date", ""),
            })

        logger.info(
            f"✅ Added {len(valid_docs)} documents to VectorStore"
        )
        return len(valid_docs)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for the most similar documents using cosine similarity.

        Args:
            query: The search query text.
            top_k: Number of results to return.

        Returns:
            List of dicts with title, url, text, source, score.
        """
        if self.embeddings is None or len(self.metadata) == 0:
            logger.warning("VectorStore is empty, no results to return.")
            return []

        top_k = min(top_k, len(self.metadata))

        # Embed the query using the fitted vectorizer
        query_vec = generate_embedding(query).reshape(1, -1)

        # Compute cosine similarity
        scores = cosine_similarity(query_vec, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "title": meta["title"],
                "url": meta["url"],
                "text": meta["text"],
                "source": meta["source"],
                "score": round(float(scores[idx]), 4),
            })

        logger.info(
            f"🔍 Search for '{query[:60]}...' → "
            f"{len(results)} results "
            f"(scores: {[r['score'] for r in results]})"
        )
        return results


def query_vectors(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """
    Legacy function for backward compatibility with rag_pipeline.

    Note: With TF-IDF approach, use VectorStore.search() directly instead.
    """
    logger.warning("query_vectors() called but no persistent store exists. "
                    "Use VectorStore.search() instead.")
    return []
