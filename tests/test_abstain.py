from app.rag.nodes.abstain import abstain


def test_abstain_sets_answer_and_clears_documents_and_sources():
    state = {
        "query": "Tell me about the weather in France.",
        "retrieved_documents": [{"chunk_id": "chunk_1"}],
        "reranked_documents": [{"chunk_id": "chunk_1", "reranker_score": 0.1}],
        "sources": [{"chunk_id": "chunk_1"}],
    }

    result = abstain(state)

    assert result["answer"]
    assert result["retrieved_documents"] == []
    assert result["reranked_documents"] == []
    assert result["sources"] == []
