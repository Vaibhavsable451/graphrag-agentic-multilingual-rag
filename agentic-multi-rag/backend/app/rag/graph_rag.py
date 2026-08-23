"""
GraphRAG: builds a lightweight knowledge graph (entities + relations)
from ingested documents and answers questions by traversing that
graph, in addition to plain vector similarity. This catches
multi-hop questions ("how is X related to Y") that pure vector
search misses.

Uses networkx for the graph itself, and stores each node's summary
text embedding in Pinecone (namespace "graph") so nodes are also
semantically searchable.
"""
import json
import uuid

import networkx as nx

from app.core.llm_router import get_llm
from app.vectorstore.pinecone_client import get_index, embed_texts

_graph = nx.DiGraph()

_EXTRACTION_PROMPT = """Extract entities and relationships from the text below.
Return ONLY valid JSON in this exact shape, nothing else:
{{
  "entities": [{{"name": "...", "type": "..."}}],
  "relations": [{{"source": "...", "target": "...", "relation": "..."}}]
}}

Text:
{text}
"""


def extract_graph_from_text(text: str) -> dict:
    llm = get_llm()
    response = llm.invoke(_EXTRACTION_PROMPT.format(text=text))
    content = response.content.strip().strip("`").removeprefix("json").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}


def ingest_document_into_graph(text: str, source: str = "unknown"):
    extracted = extract_graph_from_text(text)

    for ent in extracted.get("entities", []):
        _graph.add_node(ent["name"], type=ent.get("type", "unknown"), source=source)

    for rel in extracted.get("relations", []):
        _graph.add_edge(rel["source"], rel["target"], relation=rel.get("relation", "related_to"))

    # Also push entity nodes to Pinecone (separate namespace) so they're
    # retrievable by semantic similarity, not just exact-name graph lookup.
    entities = extracted.get("entities", [])
    if entities:
        index = get_index()
        vectors = embed_texts([f"{e['name']} ({e.get('type', '')})" for e in entities])
        payload = [
            {
                "id": f"graph-{uuid.uuid4()}",
                "values": vec,
                "metadata": {"name": e["name"], "type": e.get("type", ""), "source": source},
            }
            for e, vec in zip(entities, vectors)
        ]
        index.upsert(vectors=payload, namespace="graph")

    return extracted


def get_neighbors(entity_name: str, depth: int = 2) -> list[dict]:
    """Multi-hop traversal from an entity, used by the agentic RAG
    graph node to answer relational questions."""
    if entity_name not in _graph:
        return []

    visited = {entity_name}
    frontier = [entity_name]
    facts = []

    for _ in range(depth):
        next_frontier = []
        for node in frontier:
            for neighbor in _graph.successors(node):
                relation = _graph.edges[node, neighbor]["relation"]
                facts.append({"source": node, "relation": relation, "target": neighbor})
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier

    return facts


def export_graph_json() -> dict:
    """Used by the /graph endpoint so the React frontend can render
    the knowledge graph (nodes + edges) with react-force-graph."""
    return {
        "nodes": [{"id": n, **d} for n, d in _graph.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d} for u, v, d in _graph.edges(data=True)],
    }
