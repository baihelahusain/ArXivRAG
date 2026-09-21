from app.retrieval.reference_filter import is_reference_only


def test_normal_research_chunk_is_kept():
    document = {
        "text": (
            "Retrieval augmented generation combines retrieved passages "
            "with language model generation to improve factual grounding."
        )
    }

    assert is_reference_only(document) is False


def test_reference_only_chunk_is_filtered():
    document = {
        "text": (
            "[1] Smith A. Retrieval systems. https://example.com/a "
            "[2] Jones B. Neural ranking. doi:10.1000/example "
            "[3] Lee C. Dense retrieval. "
            "[4] Patel D. Question answering. "
            "[5] Chen E. Language models."
        )
    }

    assert is_reference_only(document) is True


def test_appendix_research_content_is_kept():
    document = {
        "text": (
            "Appendix A provides implementation details for the retrieval "
            "pipeline. The dense retriever encodes each query, BM25 handles "
            "keyword matching, and the final candidates are reranked before "
            "answer generation. See [1] for background."
        )
    }

    assert is_reference_only(document) is False
