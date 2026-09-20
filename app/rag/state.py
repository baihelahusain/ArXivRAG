from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    query: str

    rewritten_query: str

    route: str

    retrieved_documents: list[dict[str, Any]]

    reranked_documents: list[dict[str, Any]]

    relevance_grade: str

    answer: str

    sources: list[dict[str, Any]]