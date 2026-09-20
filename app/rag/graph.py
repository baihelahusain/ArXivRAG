from langgraph.graph import StateGraph, END

from app.rag.state import RAGState
from app.rag.nodes.query_router import query_router
from app.rag.nodes.query_rewriter import rewrite_query
from app.rag.nodes.retrieve import retrieve
from app.rag.nodes.rerank import rerank
from app.rag.nodes.generate_answer import generate_answer
from app.rag.nodes.query_intent import query_intent
from app.rag.nodes.abstain import abstain

RERANKER_THRESHOLD = 0.65


def route_after_router(state: RAGState):
    if state["route"] == "rewrite":
        return "rewrite"

    return "retrieve"


def check_retrieval_confidence(state: RAGState):
    documents = state.get("reranked_documents", [])

    if not documents:
        return "intent"

    max_score = documents[0].get("reranker_score", 0.0)

    print(f"Reranker max score: {max_score:.4f}")

    if max_score >= RERANKER_THRESHOLD:
        return "generate"

    return "intent"


def route_after_intent(state: RAGState):
    if state["route"] == "research_question":
        return "rewrite"

    if state["route"] == "ambiguous":
        return "rewrite"

    return "abstain"


builder = StateGraph(RAGState)

builder.add_node("query_router", query_router)
builder.add_node("rewrite", rewrite_query)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("query_intent", query_intent)
builder.add_node("generate_answer", generate_answer)
builder.add_node("abstain", abstain)

builder.set_entry_point("query_router")


builder.add_conditional_edges(
    "query_router",
    route_after_router,
    {
        "retrieve": "retrieve",
        "rewrite": "rewrite",
    },
)


builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "rerank")


builder.add_conditional_edges(
    "rerank",
    check_retrieval_confidence,
    {
        "generate": "generate_answer",
        "intent": "query_intent",
    },
)


builder.add_conditional_edges(
    "query_intent",
    route_after_intent,
    {
        "rewrite": "rewrite",
        "abstain": "abstain",
    },
)


builder.add_edge("generate_answer", END)
builder.add_edge("abstain", END)


rag_graph = builder.compile()