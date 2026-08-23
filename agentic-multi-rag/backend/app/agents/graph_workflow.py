"""
LangGraph orchestration layer. The actual decision-making logic
(routing, fused retrieval, context grading) lives in
`app/rag/agentic_rag.py` — this file just wires those functions into
a StateGraph so the flow can loop (grade -> retry retrieve) instead
of running once straight through.

Flow:
  detect_language -> route -> (chitchat -> localize)
                            -> (retrieve -> grade -> retry retrieve | generate -> localize) -> END

Small talk ("hi", "thanks") short-circuits straight to a direct reply
instead of going through retrieval — there's no document that will
ever answer a greeting, so routing it through retrieve/grade would
just waste calls and produce a "context is missing" non-answer.
"""
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from app.rag.multilingual_rag import prepare_multilingual_query, localize_answer
from app.rag.agentic_rag import (
    MAX_RETRIES,
    decide_route,
    retrieve,
    has_sufficient_context,
    widen_route,
    generate_answer,
    generate_chitchat_answer,
)


class AgentState(TypedDict):
    question: str
    lang: str
    english_question: str
    route: str
    documents: list[dict]
    graph_facts: list[dict]
    answer: str
    retries: int
    needs_retry: bool


def detect_language_node(state: AgentState) -> AgentState:
    info = prepare_multilingual_query(state["question"])
    state["lang"] = info["lang"]
    state["english_question"] = info["english"]
    state["retries"] = 0
    return state


def route_node(state: AgentState) -> AgentState:
    state["route"] = decide_route(state["english_question"])
    return state


def route_after_classify(state: AgentState) -> Literal["chitchat", "retrieve"]:
    """Pure router — small talk skips retrieval entirely."""
    return "chitchat" if state["route"] == "chitchat" else "retrieve"


def chitchat_node(state: AgentState) -> AgentState:
    state["answer"] = generate_chitchat_answer(state["english_question"])
    state["documents"] = []
    state["graph_facts"] = []
    return state


def retrieve_node(state: AgentState) -> AgentState:
    result = retrieve(state["english_question"], state["route"])
    state["documents"] = result["documents"]
    state["graph_facts"] = result["graph_facts"]
    return state


def grade_documents_node(state: AgentState) -> AgentState:
    """Decides retry vs. proceed, and does the state mutation (widen
    route, bump retries) here — inside a real node. Conditional edge
    functions (should_retry, route_after_classify) are routers only:
    LangGraph does not reliably persist mutations made inside them, so
    incrementing `retries` there caused an infinite retrieve<->grade
    loop that blew past the recursion limit instead of stopping after
    MAX_RETRIES."""
    if not has_sufficient_context(state["documents"], state["graph_facts"]) and state["retries"] < MAX_RETRIES:
        state["route"] = widen_route(state["route"])
        state["retries"] += 1
        state["needs_retry"] = True
    else:
        state["needs_retry"] = False
    return state


def should_retry(state: AgentState) -> Literal["retrieve", "generate"]:
    """Pure router — reads state, never mutates it."""
    return "retrieve" if state["needs_retry"] else "generate"


def generate_node(state: AgentState) -> AgentState:
    state["answer"] = generate_answer(state["english_question"], state["documents"], state["graph_facts"])
    return state


def localize_node(state: AgentState) -> AgentState:
    state["answer"] = localize_answer(state["answer"], state["lang"])
    return state


def build_agentic_rag_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("detect_language", detect_language_node)
    workflow.add_node("route_question", route_node)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("localize", localize_node)

    workflow.set_entry_point("detect_language")
    workflow.add_edge("detect_language", "route_question")
    workflow.add_conditional_edges(
        "route_question", route_after_classify, {"chitchat": "chitchat", "retrieve": "retrieve"}
    )
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", should_retry, {"retrieve": "retrieve", "generate": "generate"})
    workflow.add_edge("generate", "localize")
    workflow.add_edge("chitchat", "localize")
    workflow.add_edge("localize", END)

    return workflow.compile()


agentic_rag_app = build_agentic_rag_graph()


def run_agentic_rag(question: str) -> dict:
    result = agentic_rag_app.invoke({"question": question})
    return {
        "answer": result["answer"],
        "language": result["lang"],
        "route": result["route"],
        "sources": result["documents"],
        "graph_facts": result["graph_facts"],
    }
