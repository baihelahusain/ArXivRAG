from app.rag.nodes.query_router import query_router


def test_normal_research_question_routes_to_retrieve():
    state = {
        "query": "How does retrieval augmented generation reduce hallucination?"
    }

    result = query_router(state)

    assert result["route"] == "retrieve"
    assert "rewritten_query" not in result


def test_short_query_routes_to_rewrite():
    state = {"query": "RAG"}

    result = query_router(state)

    assert result["route"] == "rewrite"
    assert result["rewritten_query"] == "RAG"
