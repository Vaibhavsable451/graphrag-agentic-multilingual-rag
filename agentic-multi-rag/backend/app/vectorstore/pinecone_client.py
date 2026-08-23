"""
Pinecone vector store wrapper used by both the Agentic RAG retriever
and the GraphRAG node-embedding lookup.
"""
from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.core.embeddings import embed_query, embed_texts

_pc: Pinecone | None = None
_EMBED_DIM = 768  # matches paraphrase-multilingual-mpnet-base-v2


def get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_index():
    pc = get_client()
    existing = [i.name for i in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=_EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
        )
    return pc.Index(settings.pinecone_index_name)


def upsert_chunks(chunks: list[dict]):
    """chunks: [{"id": str, "text": str, "metadata": dict}, ...]"""
    index = get_index()
    vectors = embed_texts([c["text"] for c in chunks])
    payload = [
        {"id": c["id"], "values": vec, "metadata": {**c.get("metadata", {}), "text": c["text"]}}
        for c, vec in zip(chunks, vectors)
    ]
    index.upsert(vectors=payload)


def similarity_search(query: str, top_k: int = 5, filter: dict | None = None):
    index = get_index()
    vec = embed_query(query)
    result = index.query(vector=vec, top_k=top_k, include_metadata=True, filter=filter)
    return [
        {"id": m.id, "score": m.score, "text": m.metadata.get("text", ""), "metadata": m.metadata}
        for m in result.matches
    ]
