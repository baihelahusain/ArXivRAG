from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from app.retrieval.reference_filter import is_reference_only
from app.rag.state import RAGState


BASE_DIR = Path(__file__).resolve().parents[3]

QDRANT_COLLECTION = "research_papers"

OPENSEARCH_INDEX = "research_papers"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DENSE_TOP_K = 5
BM25_TOP_K = 5

RRF_K = 60


# Load models and clients once.
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cuda",
)

qdrant = QdrantClient(
    host="localhost",
    port=6333,
)

opensearch = OpenSearch(
    hosts=[
        {
            "host": "localhost",
            "port": 9200,
        }
    ]
)


def dense_search(query: str):
    """
    Search Qdrant using the embedding model.
    """

    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=DENSE_TOP_K,
        with_payload=True,
    )

    documents = []

    for result in results.points:

        payload = result.payload

        documents.append(
            {
                "paper_id": payload["paper_id"],
                "chunk_id": payload["chunk_id"],
                "title": payload["title"],
                "section": payload["section"],
                "page_numbers": payload["page_numbers"],
                "text": payload["text"],
                "dense_score": float(result.score),
            }
        )

    return documents


def bm25_search(query: str):
    """
    Search OpenSearch using BM25.
    """

    response = opensearch.search(
        index=OPENSEARCH_INDEX,
        body={
            "size": BM25_TOP_K,
            "query": {
                "match": {
                    "text": query
                }
            },
        },
    )

    documents = []

    for hit in response["hits"]["hits"]:

        source = hit["_source"]

        documents.append(
            {
                "paper_id": source["paper_id"],
                "chunk_id": source["chunk_id"],
                "title": source["title"],
                "section": source["section"],
                "page_numbers": source["page_numbers"],
                "text": source["text"],
                "bm25_score": float(hit["_score"]),
            }
        )

    return documents


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
):
    """
    Combine dense and BM25 rankings using RRF.
    """

    scores = {}
    documents = {}

    result_lists = [
        dense_results,
        bm25_results,
    ]

    for results in result_lists:

        for rank, document in enumerate(
            results,
            start=1,
        ):

            paper_id = document["paper_id"]

            if paper_id not in scores:

                scores[paper_id] = 0.0
                documents[paper_id] = document

            scores[paper_id] += (
                1 / (RRF_K + rank)
            )

    fused_documents = []

    for paper_id, score in scores.items():

        document = documents[paper_id].copy()

        document["rrf_score"] = score

        fused_documents.append(document)

    fused_documents.sort(
        key=lambda document: document["rrf_score"],
        reverse=True,
    )

    return fused_documents


def retrieve(state: RAGState) -> RAGState:
    """
    Execute dense + BM25 hybrid retrieval.
    """

    query = state.get(
        "rewritten_query",
        state["query"],
    )

    print(f"\nRetrieving for query: {query}")

    dense_results = dense_search(query)

    print(
        f"Dense results: {len(dense_results)}"
    )

    bm25_results = bm25_search(query)

    print(
        f"BM25 results: {len(bm25_results)}"
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
    )

    print(
        f"RRF candidates: {len(fused_results)}"
    )

    filtered_results = [
        document
        for document in fused_results
        if not is_reference_only(document)
    ]

    print(
        f"After reference filter: {len(filtered_results)}"
    )

    state["retrieved_documents"] = filtered_results

    return state