import logging
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from app.services.embedding_service import generate_embedding, generate_embeddings

logger = logging.getLogger(__name__)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _try_nltk_sentence_split(text: str) -> Optional[List[str]]:
    try:
        import nltk
        from nltk.tokenize import sent_tokenize

        # Ensure punkt exists; if not, fall back to regex splitter (no download in prod paths)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            return None

        return sent_tokenize(text)
    except Exception:
        return None


def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    nltk_sents = _try_nltk_sentence_split(cleaned)
    if nltk_sents:
        sents = nltk_sents
    else:
        sents = _SENTENCE_SPLIT_RE.split(cleaned)

    # Basic cleanup / guardrails
    out: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) < 20:
            continue
        out.append(s)
    return out


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns cosine similarity between each row of a (N,D) and b (D,) => (N,)
    """
    if a.size == 0:
        return np.zeros((0,), dtype=np.float32)

    b = b.astype(np.float32)
    a = a.astype(np.float32)

    a_norm = np.linalg.norm(a, axis=1) + 1e-8
    b_norm = np.linalg.norm(b) + 1e-8
    sims = (a @ b) / (a_norm * b_norm)
    return sims.astype(np.float32)


def extract_relevant_sentences(claim: str, article_text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Split article_text into sentences, compute similarity to claim, and return top_n.

    Returns list items:
      {"sentence": str, "similarity": float}
    """
    sentences = split_into_sentences(article_text)
    logger.info(
        "EvidenceExtractionStart claim=%r sentences_scanned=%s",
        claim[:120],
        len(sentences),
    )

    if not sentences:
        return []

    claim_vec = generate_embedding(claim)
    sent_vecs = generate_embeddings(sentences)
    sims = _cosine_sim_matrix(sent_vecs, claim_vec)

    ranked_idx = np.argsort(-sims)[: max(0, int(top_n))]
    selected: List[Dict[str, Any]] = []
    for idx in ranked_idx:
        selected.append(
            {
                "sentence": sentences[int(idx)],
                "similarity": float(sims[int(idx)]),
            }
        )

    logger.info(
        "EvidenceExtractionSelected claim=%r top_selected=%s",
        claim[:120],
        len(selected),
    )
    return selected


def dedupe_by_sentence_similarity(
    sentence_items: List[Dict[str, Any]],
    threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    Deduplicate sentence items by sentence-to-sentence cosine similarity.

    Uses embeddings; keeps the first occurrence (assumes caller already sorted by relevance).
    """
    if not sentence_items:
        return []

    sentences = [it.get("sentence", "") for it in sentence_items]
    vecs = generate_embeddings(sentences)
    if vecs.size == 0:
        return sentence_items

    norms = np.linalg.norm(vecs, axis=1) + 1e-8
    kept: List[Dict[str, Any]] = []
    kept_vecs: List[np.ndarray] = []
    kept_norms: List[float] = []

    for i, item in enumerate(sentence_items):
        v = vecs[i].astype(np.float32)
        vn = float(norms[i])
        is_dup = False
        for kv, kn in zip(kept_vecs, kept_norms):
            sim = float((v @ kv) / (vn * kn))
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
            kept_vecs.append(v)
            kept_norms.append(vn)

    logger.info(
        "EvidenceExtractionDedup threshold=%s before=%s after=%s",
        threshold,
        len(sentence_items),
        len(kept),
    )
    return kept

