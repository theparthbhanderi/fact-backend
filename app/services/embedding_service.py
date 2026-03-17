"""
Embedding Service (Dense Vector Embeddings for RAG).

Uses sentence-transformers to generate robust, dense vector embeddings
for factual claim matching and evidence retrieval. 

The interface remains identical for callers:
    generate_embedding(text)      → numpy array (384,)
    generate_embeddings([texts])  → numpy array (N, 384)
"""

import logging
import numpy as np
import threading
from app.services.cache_service import disk_cache, DiskCache

logger = logging.getLogger(__name__)

_embedder = None
_embedder_lock = threading.Lock()
_embedder_load_attempted = False

EMBEDDING_DIM = 384

def create_vectorizer(corpus: list[str]):
    """No-op for backward compatibility. Sentence Transformers don't need fitting."""
    logger.info("create_vectorizer called - ignoring as dense embeddings are pre-trained.")
    return None

def get_embedding_dim() -> int:
    """Return embedding dimension (384 for all-MiniLM-L6-v2)."""
    return EMBEDDING_DIM


def _get_embedder():
    """
    Lazy-load SentenceTransformer to avoid high memory usage at import time.
    Returns None if model can't be loaded.
    """
    global _embedder, _embedder_load_attempted
    if _embedder is not None:
        return _embedder
    if _embedder_load_attempted:
        return None

    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        if _embedder_load_attempted:
            return None
        _embedder_load_attempted = True
        try:
            # Reduce CPU/memory pressure on small instances
            try:
                import torch

                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

            from sentence_transformers import SentenceTransformer

            logger.info("🧠 Lazy-loading SentenceTransformer model: all-MiniLM-L6-v2")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            return _embedder
        except ImportError:
            logger.warning("⚠️ sentence-transformers not installed or loading failed.")
            return None
        except Exception as e:
            logger.error("❌ Failed to load SentenceTransformer: %s", e)
            return None

def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a dense vector embedding for a single text string.

    Args:
        text: The text to encode.

    Returns:
        A dense numpy array representing the text (384 dimensions).
    """
    if not text or not text.strip():
        logger.warning("Empty text, returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    embedder = _get_embedder()
    if embedder is None:
        logger.error("Embedder is missing, returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    try:
        payload = {"t": text.strip(), "dim": EMBEDDING_DIM, "model": "all-MiniLM-L6-v2"}
        key = DiskCache.make_key("embedding_single", payload)
        cached = disk_cache.get(key)
        if cached and isinstance(cached, dict) and isinstance(cached.get("v"), list):
            vec = np.array(cached["v"], dtype=np.float32)
        else:
            vec = embedder.encode(text, convert_to_numpy=True).astype(np.float32)
            disk_cache.set(key, {"v": vec.tolist()}, ttl_s=24 * 3600)
        if vec.shape[0] != EMBEDDING_DIM:
            # fallback if cache corrupted
            vec = embedder.encode(text, convert_to_numpy=True).astype(np.float32)
        return vec.astype(np.float32)
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

def generate_embeddings(text_list: list[str]) -> np.ndarray:
    """
    Generate dense vector embeddings for a batch of texts.

    Args:
        text_list: List of text strings to encode.

    Returns:
        A numpy array of shape (N, 384).
    """
    if not text_list:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    embedder = _get_embedder()
    if embedder is None:
        logger.error("Embedder is missing, returning empty array.")
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    cleaned = [t if t and t.strip() else " " for t in text_list]
    logger.info(f"🔢 Generating dense vectors for {len(cleaned)} texts...")
    
    try:
        vectors: list[np.ndarray] = []
        to_compute: list[str] = []
        idx_map: list[int] = []

        # Per-item cache
        for i, t in enumerate(cleaned):
            payload = {"t": t.strip(), "dim": EMBEDDING_DIM, "model": "all-MiniLM-L6-v2"}
            key = DiskCache.make_key("embedding_single", payload)
            hit = disk_cache.get(key)
            if hit and isinstance(hit, dict) and isinstance(hit.get("v"), list) and len(hit.get("v")) == EMBEDDING_DIM:
                vectors.append(np.array(hit["v"], dtype=np.float32))
            else:
                vectors.append(None)  # type: ignore
                to_compute.append(t)
                idx_map.append(i)

        if to_compute:
            computed = embedder.encode(to_compute, convert_to_numpy=True).astype(np.float32)
            for j, i in enumerate(idx_map):
                vec = computed[j].astype(np.float32)
                vectors[i] = vec  # type: ignore
                payload = {"t": cleaned[i].strip(), "dim": EMBEDDING_DIM, "model": "all-MiniLM-L6-v2"}
                key = DiskCache.make_key("embedding_single", payload)
                disk_cache.set(key, {"v": vec.tolist()}, ttl_s=24 * 3600)

        arr = np.vstack([v if v is not None else np.zeros(EMBEDDING_DIM, dtype=np.float32) for v in vectors]).astype(np.float32)
        logger.info(f"✅ Generated {arr.shape[0]} dense vectors")
        return arr
    except Exception as e:
        logger.error(f"❌ Batch embedding generation failed: {e}")
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
