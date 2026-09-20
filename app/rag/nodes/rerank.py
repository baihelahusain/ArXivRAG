from sentence_transformers import CrossEncoder

from app.rag.state import RAGState


RERANKER_MODEL = "BAAI/bge-reranker-base"

FINAL_TOP_K = 5


# Load the reranker once.
reranker = CrossEncoder(
    RERANKER_MODEL,
    device="cuda",
)


def rerank(state: RAGState) -> RAGState:
    """
    Rerank the documents returned by hybrid retrieval.
    """

    query = state.get(
        "rewritten_query",
        state["query"],
    )

    documents = state.get(
        "retrieved_documents",
        [],
    )

    if not documents:
        state["reranked_documents"] = []
        return state

    pairs = []

    for document in documents:
        pairs.append(
            [
                query,
                document["text"],
            ]
        )

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked_documents = []

    for document, score in zip(
        documents,
        scores,
    ):
        new_document = document.copy()

        new_document["reranker_score"] = float(
            score
        )

        reranked_documents.append(
            new_document
        )

    reranked_documents.sort(
        key=lambda document: document["reranker_score"],
        reverse=True,
    )

    state["reranked_documents"] = (
        reranked_documents[:FINAL_TOP_K]
    )

    return state