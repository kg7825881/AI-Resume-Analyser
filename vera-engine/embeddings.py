"""
embeddings.py — embedding-based similarity utilities.

Split out of matcher.py when the default scoring pipeline moved to BM25
retrieval + LLM evidence judgment (see matcher.py's module docstring). The
functions here are NOT used by scorer.py / matcher.py anymore — kept around
so calibrate_embeddings.py still works, and so an embedding-based semantic
stage stays a small, drop-in addition later (e.g. an optional third stage
that pre-filters candidates before the more expensive LLM judge call) rather
than something that has to be rebuilt from scratch if it turns out useful.
"""

import math
import threading

import ollama

EMBED_MODEL = "nomic-embed-text"

_embedding_cache: dict[str, list[float]] = {}
_cache_lock = threading.Lock()


def get_embedding(text: str) -> list[float]:
    """Generates (or returns a cached) vector embedding for a string via the local embedding model."""
    with _cache_lock:
        cached = _embedding_cache.get(text)
    if cached is not None:
        return cached

    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    vec = response["embedding"]

    with _cache_lock:
        _embedding_cache[text] = vec
    return vec


def cache_stats() -> dict:
    """Useful for confirming the cache is actually being hit during a batch run."""
    with _cache_lock:
        return {"cached_strings": len(_embedding_cache)}


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)
