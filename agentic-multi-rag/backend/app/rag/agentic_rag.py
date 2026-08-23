"""
Agentic RAG: the decision-making core that picks HOW to retrieve
for a given question, and self-corrects if the first attempt comes
back empty — as opposed to plain RAG, which always does one fixed
vector lookup regardless of the question.

This module holds the reusable logic (route classification, fused
retrieval, context grading). `app/agents/graph_workflow.py` wires
these functions into a LangGraph StateGraph so the flow can loop
(re-route -> retrieve again) instead of running once, straight
through.

Sits alongside graph_rag.py (graph traversal) and
multilingual_rag.py (language handling) — this module is what
decides *when* to use each of them.
"""
from app.core.llm_router import get_llm
from app.rag.graph_rag import get_neighbors
from app.vectorstore.pinecone_client import similarity_search

MAX_RETRIES = 2

_ROUTE_PROMPT = """Classify the question into exactly one label:
'chitchat' for greetings, thanks, or small talk with no factual question,
'vector' for a fact-lookup question,
'graph' for a question about relationships/connections between entities,
'both' if unsure. Respond with only the label.

Question: {question}"""

_VALID_ROUTES = ("chitchat", "vector", "graph", "both")


def decide_route(question: str) -> str:
    """LLM call that decides whether this question is small talk, needs
    plain vector retrieval, graph traversal (relational/multi-hop), or
    both. Small talk skips retrieval entirely — there's no document
    that will ever answer "hi"."""
    llm = get_llm()
    label = llm.invoke(_ROUTE_PROMPT.format(question=question)).content.strip().lower()
    return label if label in _VALID_ROUTES else "both"


def generate_chitchat_answer(question: str) -> str:
    """Direct, context-free reply for greetings/small talk — no
    retrieval involved, so it never says 'context is missing' for a
    'hi'."""
    llm = get_llm()
    prompt = (
        "You are a helpful RAG assistant for a knowledge base. Reply "
        "briefly and naturally to this small talk / greeting, and "
        "invite the person to ask a question about the ingested "
        f"knowledge base.\n\nMessage: {question}"
    )
    return llm.invoke(prompt).content


def _guess_entities(question: str, limit: int = 3) -> list[str]:
    """Cheap heuristic entity guesser (capitalized tokens) used to seed
    graph traversal without a second LLM call. Good enough for a demo;
    swap for a proper NER pass if you need higher recall."""
    return [w.strip("?,.") for w in question.split() if w[:1].isupper()][:limit]


def retrieve(question: str, route: str, top_k: int = 5) -> dict:
    """Fused retrieval: runs vector search and/or graph traversal
    depending on the chosen route, and returns both result sets."""
    documents: list[dict] = []
    graph_facts: list[dict] = []

    if route in ("vector", "both"):
        documents = similarity_search(question, top_k=top_k)

    if route in ("graph", "both"):
        for entity in _guess_entities(question):
            graph_facts.extend(get_neighbors(entity))

    return {"documents": documents, "graph_facts": graph_facts}


def has_sufficient_context(documents: list[dict], graph_facts: list[dict]) -> bool:
    """LLM-as-judge stand-in: cheap check for now (non-empty result
    sets); swap in an LLM relevance grader here if recall needs to be
    tighter than 'retrieval returned something'."""
    return bool(documents) or bool(graph_facts)


def widen_route(current_route: str) -> str:
    """On a failed grade, always widen to 'both' rather than narrowing —
    the point of self-correction is to cast a bigger net, not a smaller one."""
    return "both"


def self_correcting_retrieve(question: str, route: str | None = None) -> dict:
    """Runs the full agentic retrieval loop standalone (outside
    LangGraph) — decide route -> retrieve -> grade -> retry up to
    MAX_RETRIES, widening the route each time. Useful for testing the
    agentic behavior directly, or for callers that don't need the
    full StateGraph (e.g. the MCP server, a notebook)."""
    route = route or decide_route(question)
    attempt = 0
    result = {"documents": [], "graph_facts": []}

    while attempt <= MAX_RETRIES:
        result = retrieve(question, route)
        if has_sufficient_context(result["documents"], result["graph_facts"]):
            break
        route = widen_route(route)
        attempt += 1

    return {**result, "route": route, "retries": attempt}


def generate_answer(question: str, documents: list[dict], graph_facts: list[dict]) -> str:
    """Grounded generation step — only answers from what was retrieved,
    and says so explicitly when context falls short."""
    llm = get_llm()
    context_chunks = "\n".join(f"- {d['text']}" for d in documents)
    graph_chunks = "\n".join(
        f"- {f['source']} --{f['relation']}--> {f['target']}" for f in graph_facts
    )
    prompt = f"""Answer the question using ONLY the context below. If the context is
insufficient, say what's missing instead of guessing.

Vector context:
{context_chunks or "(none)"}

Graph facts:
{graph_chunks or "(none)"}

Question: {question}
Answer:"""
    return llm.invoke(prompt).content
