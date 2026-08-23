"""
MCP (Model Context Protocol) server that exposes this project's RAG
capabilities as standard MCP tools, so any MCP-compatible client
(Claude Desktop, another agent, etc.) can call them directly instead
of only being usable through this project's own React frontend.

Run standalone with: python -m app.mcp.mcp_server
"""
from mcp.server.fastmcp import FastMCP

from app.agents.graph_workflow import run_agentic_rag
from app.rag.graph_rag import export_graph_json, ingest_document_into_graph
from app.vectorstore.pinecone_client import upsert_chunks

mcp = FastMCP("agentic-multi-rag")


@mcp.tool()
def ask_rag(question: str) -> dict:
    """Answer a question using the agentic multilingual GraphRAG pipeline."""
    return run_agentic_rag(question)


@mcp.tool()
def ingest_text(text: str, source: str = "mcp-client") -> dict:
    """Ingest a document into both the vector store and the knowledge graph."""
    chunk_id = f"{source}-{abs(hash(text))}"
    upsert_chunks([{"id": chunk_id, "text": text, "metadata": {"source": source}}])
    graph_extract = ingest_document_into_graph(text, source=source)
    return {"chunk_id": chunk_id, "entities_found": len(graph_extract.get("entities", []))}


@mcp.tool()
def get_knowledge_graph() -> dict:
    """Return the current knowledge graph as nodes/edges for inspection or visualization."""
    return export_graph_json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
