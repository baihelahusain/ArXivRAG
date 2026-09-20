from app.rag.state import RAGState


RELEVANCE_THRESHOLD = 0.50


def grade_documents(state: RAGState) -> RAGState:
    documents = state.get("reranked_documents", [])

    if not documents:
        state["relevance_grade"] = "not_relevant"
        return state

    relevant_documents = []

    for document in documents:
        score = document.get("reranker_score", 0.0)

        if score >= RELEVANCE_THRESHOLD:
            relevant_documents.append(document)

    state["reranked_documents"] = relevant_documents

    if relevant_documents:
        state["relevance_grade"] = "relevant"
    else:
        state["relevance_grade"] = "not_relevant"

    return state