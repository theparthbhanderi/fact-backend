"""
Embedding Service (Lightweight TF-IDF).

Uses scikit-learn's TF-IDF vectorizer instead of heavy
sentence-transformers + PyTorch to keep memory under 512MB
on free-tier hosting (Render).

The interface remains identical so no other code needs changes:
    generate_embedding(text)      → numpy array
    generate_embeddings([texts])  → numpy array (N, dim)

Note: TF-IDF vectors are sparse and variable-dimension.
      We convert them to dense arrays for FAISS-free cosine search.
"""

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ── Global vectorizer (fitted per-request in VectorStore) ──────────────
# Unlike sentence-transformers, TF-IDF must be fitted on a corpus first.
# We expose a simple wrapper that VectorStore will call.

from typing import Optional

_vectorizer: Optional[TfidfVectorizer] = None


def create_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the given corpus and cache it globally.

    Args:
        corpus: List of text documents to build vocabulary from.

    Returns:
        The fitted TfidfVectorizer instance.
    """
    global _vectorizer
    _vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    _vectorizer.fit(corpus)
    logger.info(f"✅ TF-IDF vectorizer fitted on {len(corpus)} documents "
                f"(vocab size: {len(_vectorizer.vocabulary_)})")
    return _vectorizer


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a TF-IDF vector for a single text string.

    Args:
        text: The text to encode.

    Returns:
        A dense numpy array representing the text.
    """
    if _vectorizer is None:
        logger.warning("Vectorizer not fitted yet, returning empty array.")
        return np.zeros(1, dtype=np.float32)

    if not text or not text.strip():
        dim = len(_vectorizer.vocabulary_)
        return np.zeros(dim, dtype=np.float32)

    vec = _vectorizer.transform([text]).toarray().astype(np.float32)
    return vec[0]


def generate_embeddings(text_list: list[str]) -> np.ndarray:
    """
    Generate TF-IDF vectors for a batch of texts.

    Args:
        text_list: List of text strings to encode.

    Returns:
        A numpy array of shape (N, vocab_size).
    """
    if _vectorizer is None:
        logger.warning("Vectorizer not fitted yet, returning empty array.")
        return np.empty((0, 1), dtype=np.float32)

    if not text_list:
        dim = len(_vectorizer.vocabulary_)
        return np.empty((0, dim), dtype=np.float32)

    cleaned = [t if t and t.strip() else " " for t in text_list]
    logger.info(f"🔢 Generating TF-IDF vectors for {len(cleaned)} texts...")
    embeddings = _vectorizer.transform(cleaned).toarray().astype(np.float32)
    logger.info(f"✅ Generated {len(embeddings)} vectors")
    return embeddings


# Expose a dynamic EMBEDDING_DIM (used by vector_store)
def get_embedding_dim() -> int:
    """Return current embedding dimension (vocab size)."""
    if _vectorizer is None:
        return 0
    return len(_vectorizer.vocabulary_)


# Keep backward compat — some imports reference EMBEDDING_DIM directly
EMBEDDING_DIM = 384  # placeholder, actual dim set at runtime
