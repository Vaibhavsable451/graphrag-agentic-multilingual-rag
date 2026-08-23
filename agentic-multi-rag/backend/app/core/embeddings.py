"""
Embeddings for the vector store.

Uses a local multilingual sentence-transformers model so the same
embedding space covers English, Hindi, Marathi, Spanish, etc. —
this is what makes Multilingual RAG possible without extra API cost.
"""
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # 768-dim, 50+ languages
_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return get_embedder().encode([text], normalize_embeddings=True)[0].tolist()
