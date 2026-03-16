"""
Embedding Service.

Converts text passages into dense vector embeddings using a
Sentence-Transformer model (all-MiniLM-L6-v2).

The model is loaded once at module level and reused across
all requests to avoid expensive reloading.

Usage:
    from app.services.embedding_service import generate_embedding, generate_embeddings

    vec = generate_embedding("some text")          # single text
    vecs = generate_embeddings(["text1", "text2"]) # batch
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

# ── Load model once at import time ─────────────────────────────────────
# all-MiniLM-L6-v2 produces 384-dimensional embeddings.
# It is fast, lightweight, and has high semantic similarity accuracy.
logger.info(f"⏳ Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
logger.info(f"✅ Embedding model loaded: {settings.EMBEDDING_MODEL_NAME}")

# Embedding dimension (384 for all-MiniLM-L6-v2)
EMBEDDING_DIM = _model.get_sentence_embedding_dimension()


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a vector embedding for a single text string.

    Args:
        text: The text to encode.

    Returns:
        A numpy array of shape (384,) representing the text embedding.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to generate_embedding, returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    embedding = _model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype(np.float32)


def generate_embeddings(text_list: list[str]) -> np.ndarray:
    """
    Generate vector embeddings for a batch of texts.

    Uses batch encoding for efficiency. Empty texts receive
    zero vectors to maintain index alignment.

    Args:
        text_list: List of text strings to encode.

    Returns:
        A numpy array of shape (N, 384) where N = len(text_list).
    """
    if not text_list:
        logger.warning("Empty text list provided to generate_embeddings.")
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    # Replace empty strings with a placeholder to avoid model errors
    cleaned = [t if t and t.strip() else " " for t in text_list]

    logger.info(f"🔢 Generating embeddings for {len(cleaned)} texts...")
    embeddings = _model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    logger.info(f"✅ Generated {len(embeddings)} embeddings (dim={EMBEDDING_DIM})")
    return embeddings
