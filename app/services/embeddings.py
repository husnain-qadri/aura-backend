"""Specter2 embedding service for section-level similarity."""

import numpy as np
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

_model: Any = None


def _get_model() -> Any:
    global _model
    if _model is None and HAS_SENTENCE_TRANSFORMERS:
        from app.config import SPECTER_MODEL_NAME
        _model = SentenceTransformer(SPECTER_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    if model is None:
        return [[0.0] * 768 for _ in texts]
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = float(np.dot(va, vb))
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norm < 1e-9:
        return 0.0
    return dot / norm
