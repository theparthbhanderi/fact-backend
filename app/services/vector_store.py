"""
Vector Store Service (Persistent RAG with FAISS).

Uses FAISS for persistent, high-performance similarity search.
Stores vectors in 'storage/faiss_index.bin' and metadata in 'storage/metadata.json'.
"""

import os
import json
import logging
import numpy as np
import faiss

from app.services.embedding_service import generate_embedding, generate_embeddings, get_embedding_dim

logger = logging.getLogger(__name__)

STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.bin")
META_PATH = os.path.join(STORAGE_DIR, "faiss_metadata.json")

class VectorStore:
    """
    Persistent vector store using FAISS + JSON metadata.
    """
    def __init__(self):
        self.dim = get_embedding_dim() or 384
        self.metadata = []
        self.index = None
        
        if not os.path.exists(STORAGE_DIR):
            os.makedirs(STORAGE_DIR, exist_ok=True)
            
        self._load()

    def _load(self):
        """Load index and metadata from disk if they exist, else create new."""
        try:
            if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
                self.index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"📦 Loaded persistent FAISS index with {len(self.metadata)} documents.")
            else:
                logger.info("📦 Creating new FAISS index.")
                self.index = faiss.IndexFlatL2(self.dim)
                self.metadata = []
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index: {e}. Starting fresh.")
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []

    def _save(self):
        """Save index and metadata to disk."""
        try:
            faiss.write_index(self.index, INDEX_PATH)
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info("💾 Saved FAISS index and metadata to disk.")
        except Exception as e:
            logger.error(f"❌ Failed to save FAISS index: {e}")

    def add_documents(self, documents: list[dict]) -> int:
        """
        Embed and add documents to the FAISS store persistently.
        Deduplicates based on URL.
        """
        if not documents:
            return 0

        valid_docs = [d for d in documents if d.get("text", "").strip()]
        if not valid_docs:
            return 0

        # Optional deduplication against existing
        existing_urls = {m.get("url") for m in self.metadata if m.get("url")}
        new_docs = [d for d in valid_docs if d.get("url") not in existing_urls and "url" in d]
        
        # If all were duplicates or lacking URLs, skip adding nothing
        if not new_docs:
            logger.info("No new unique documents to add to FAISS.")
            return 0

        texts = [d["text"] for d in new_docs]
        embeddings = generate_embeddings(texts)
        
        if embeddings.shape[0] == 0:
            return 0

        # FAISS expects float32 arrays natively
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self.index.add(embeddings)
        
        # Add metadata
        for doc in new_docs:
            self.metadata.append({
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "text": doc.get("text", ""),
                "source": doc.get("source", ""),
                "summary": doc.get("summary", ""),
                "publish_date": doc.get("publish_date", ""),
            })

        self._save()
        logger.info(f"✅ Added {len(new_docs)} documents to persistent VectorStore.")
        return len(new_docs)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search FAISS index for most similar documents."""
        if self.index.ntotal == 0:
            logger.warning("VectorStore is empty.")
            return []

        top_k = min(top_k, self.index.ntotal)
        
        query_vec = generate_embedding(query).reshape(1, -1).astype(np.float32)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            # FAISS L2 distance: lower is better. We invert it to normalize roughly 0-1
            score = round(1.0 / (1.0 + float(dist)), 4)
            
            results.append({
                "title": meta["title"],
                "url": meta["url"],
                "text": meta["text"],
                "source": meta["source"],
                "score": score,
            })

        logger.info(f"🔍 Search for '{query[:60]}...' yielded {len(results)} matches.")
        return results

def query_vectors(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """Legacy support."""
    logger.warning("query_vectors() called. Use VectorStore.search() directly.")
    return []
