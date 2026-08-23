import uuid

from fastapi import APIRouter

from app.agents.graph_workflow import run_agentic_rag
from app.api.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from app.rag.graph_rag import export_graph_json, ingest_document_into_graph
from app.vectorstore.pinecone_client import upsert_chunks

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_agentic_rag(req.question)
    return ChatResponse(**result)


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    chunk_id = str(uuid.uuid4())
    upsert_chunks([{"id": chunk_id, "text": req.text, "metadata": {"source": req.source}}])
    graph_extract = ingest_document_into_graph(req.text, source=req.source)
    return IngestResponse(chunk_id=chunk_id, entities_found=len(graph_extract.get("entities", [])))


@router.get("/graph")
def graph():
    return export_graph_json()


@router.get("/health")
def health():
    return {"status": "ok"}
