from app.rag.state import RAGState


def abstain(state: RAGState) -> RAGState:
    state["answer"] = (
        "I could not find sufficiently relevant research-paper evidence "
        "to answer this question."
    )

    # Do not expose rejected retrieval results as sources/documents.
    state["reranked_documents"] = []
    state["retrieved_documents"] = []
    state["sources"] = []

    return state