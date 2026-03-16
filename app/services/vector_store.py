"""
Vector Store Service.

Manages an in-memory FAISS index for storing and querying
document embeddings with associated metadata.

Architecture:
    - FAISS IndexFlatIP (inner product = cosine similarity on
      normalized vectors) for fast exact search.
    - Metadata (title, url, source, text) stored in a parallel
      Python list, indexed by FAISS row ID.
    - A new VectorStore instance is created per request to keep
      searches isolated. For persistence across requests, the
      class can be extended to save/load from disk.

Usage:
    from app.services.vector_store import VectorStore

    store = VectorStore()
    store.add_documents(articles)
    results = store.search("query text", top_k=3)
"""

import logging
import numpy as np
import faiss

from app.services.embedding_service import generate_embedding, generate_embeddings, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-backed vector store for document embeddings.

    Stores embeddings in a flat inner-product index (cosine similarity
    on L2-normalized vectors) and maintains parallel metadata.

    Attributes:
        index: The FAISS index.
        metadata: List of metadata dicts aligned with FAISS row IDs.
    """

    def __init__(self):
        """Initialize an empty FAISS index with dimension 384."""
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.metadata: list[dict] = []
        logger.info(f"📦 Initialized FAISS index (dim={EMBEDDING_DIM})")

    def add_documents(self, documents: list[dict]) -> int:
        """
        Embed and add documents to the FAISS index.

        Each document dict should contain at minimum:
            - text (str): The content to embed.
            - title (str): Article headline.
            - url (str): Article URL.
            - source (str): Publisher name.

        Optionally:
            - summary (str): Article summary.
            - publish_date (str): Publication date.

        Args:
            documents: List of document dicts.

        Returns:
            Number of documents successfully added.
        """
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return 0

        # Filter out documents with no meaningful text
        valid_docs = [d for d in documents if d.get("text", "").strip()]
        if not valid_docs:
            logger.warning("All documents have empty text, nothing to add.")
            return 0

        # Extract texts and generate embeddings in batch
        texts = [d["text"] for d in valid_docs]
        embeddings = generate_embeddings(texts)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata (everything except the raw text for search results)
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
            f"✅ Added {len(valid_docs)} documents to FAISS index "
            f"(total: {self.index.ntotal})"
        )
        return len(valid_docs)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search the FAISS index for the most similar documents.

        Embeds the query, performs inner-product search (cosine
        similarity on normalized vectors), and returns the top-k
        most relevant documents with their similarity scores.

        Args:
            query: The search query text.
            top_k: Number of results to return (default 3).

        Returns:
            A list of dicts, each containing:
                - title (str): Article headline.
                - url (str): Article URL.
                - text (str): Article text.
                - source (str): Publisher name.
                - score (float): Cosine similarity score (0–1).
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, no results to return.")
            return []

        # Clamp top_k to the number of stored documents
        top_k = min(top_k, self.index.ntotal)

        # Embed the query
        query_embedding = generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "title": meta["title"],
                "url": meta["url"],
                "text": meta["text"],
                "source": meta["source"],
                "score": round(float(score), 4),
            })

        logger.info(
            f"🔍 FAISS search for '{query[:60]}...' → "
            f"{len(results)} results "
            f"(scores: {[r['score'] for r in results]})"
        )
        return results
