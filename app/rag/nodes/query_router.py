from app.rag.state import RAGState


def query_router(state: RAGState) -> RAGState:
    """
    Decide whether the query can go directly to retrieval
    or should be rewritten first.
    """

    query = state["query"].strip()

    # Very short queries usually need more context.
    if len(query.split()) < 4:
        state["rewritten_query"] = query
        state["route"] = "rewrite"
    else:
        state["route"] = "retrieve"

    return state