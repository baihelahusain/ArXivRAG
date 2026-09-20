import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from opensearchpy import OpenSearch

from app.retrieval.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

QUERY_FILE = BASE_DIR / "evaluation" / "queries.json"
RESULT_FILE = BASE_DIR / "evaluation" / "results" / "reranker_results.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "research_papers"

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_INDEX = "research_papers"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

DENSE_TOP_K = 5
BM25_TOP_K = 5
FINAL_TOP_K = 10

RRF_K = 60


# ---------------------------------------------------------
# RRF
# ---------------------------------------------------------

def reciprocal_rank_fusion(results_lists, k=60):
    scores = {}
    documents = {}

    for results in results_lists:
        for rank, document in enumerate(results, start=1):
            paper_id = document["paper_id"]

            if paper_id not in scores:
                scores[paper_id] = 0.0
                documents[paper_id] = document

            scores[paper_id] += 1 / (k + rank)

    fused = []

    for paper_id, score in scores.items():
        document = documents[paper_id].copy()
        document["rrf_score"] = score
        fused.append(document)

    fused.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return fused


# ---------------------------------------------------------
# Dense Retrieval
# ---------------------------------------------------------

def dense_search(client, model, query, top_k=5):
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
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
                "text": payload["text"],
                "score": float(result.score),
            }
        )

    return documents


# ---------------------------------------------------------
# BM25 Retrieval
# ---------------------------------------------------------

def bm25_search(client, query, top_k=5):
    response = client.search(
        index=OPENSEARCH_INDEX,
        body={
            "size": top_k,
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
                "text": source["text"],
                "score": float(hit["_score"]),
            }
        )

    return documents


# ---------------------------------------------------------
# Remove duplicate papers
# ---------------------------------------------------------

def unique_papers(documents):
    unique = {}

    for document in documents:
        paper_id = document["paper_id"]

        if paper_id not in unique:
            unique[paper_id] = document

    return list(unique.values())


# ---------------------------------------------------------
# Reranking
# ---------------------------------------------------------

def rerank_documents(reranker, query, documents, top_k=10):
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

    reranked = []

    for document, score in zip(documents, scores):
        item = document.copy()
        item["reranker_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    print("Loading evaluation queries...")

    with open(QUERY_FILE, "r", encoding="utf-8") as file:
        queries = json.load(file)

    print(f"Loaded {len(queries)} evaluation queries")

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cuda",
    )

    print("Embedding model loaded")

    print("\nLoading BGE reranker...")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="cuda",
    )

    print("Reranker loaded")

    print("\nConnecting to Qdrant...")

    qdrant = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    print("Connected to Qdrant")

    print("\nConnecting to OpenSearch...")

    opensearch = OpenSearch(
        hosts=[
            {
                "host": OPENSEARCH_HOST,
                "port": OPENSEARCH_PORT,
            }
        ]
    )

    print("Connected to OpenSearch")

    all_results = []

    for query_number, item in enumerate(queries, start=1):

        query_id = item["query_id"]
        query = item["query"]
        relevant_papers = set(item["relevant_paper_ids"])

        print("\n" + "=" * 70)
        print(f"Query {query_number}/{len(queries)}")
        print(f"{query_id}: {query}")

        # ---------------------------------------------
        # Dense retrieval
        # ---------------------------------------------

        dense_results = dense_search(
            qdrant,
            embedding_model,
            query,
            DENSE_TOP_K,
        )

        # ---------------------------------------------
        # BM25 retrieval
        # ---------------------------------------------

        bm25_results = bm25_search(
            opensearch,
            query,
            BM25_TOP_K,
        )

        # ---------------------------------------------
        # RRF
        # ---------------------------------------------

        fused_results = reciprocal_rank_fusion(
            [
                dense_results,
                bm25_results,
            ],
            k=RRF_K,
        )

        # ---------------------------------------------
        # Remove duplicate papers
        # ---------------------------------------------

        candidates = unique_papers(fused_results)

        print(f"RRF candidates: {len(candidates)}")

        # ---------------------------------------------
        # Reranking
        # ---------------------------------------------

        reranked_results = rerank_documents(
            reranker,
            query,
            candidates,
            FINAL_TOP_K,
        )

        retrieved_papers = [
            result["paper_id"]
            for result in reranked_results
        ]

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        metrics = {
            "precision_at_5": precision_at_k(
                retrieved_papers,
                relevant_papers,
                5,
            ),
            "recall_at_5": recall_at_k(
                retrieved_papers,
                relevant_papers,
                5,
            ),
            "precision_at_10": precision_at_k(
                retrieved_papers,
                relevant_papers,
                10,
            ),
            "recall_at_10": recall_at_k(
                retrieved_papers,
                relevant_papers,
                10,
            ),
            "mrr": mrr(
                retrieved_papers,
                relevant_papers,
            ),
            "ndcg_at_5": ndcg_at_k(
                retrieved_papers,
                relevant_papers,
                5,
            ),
            "ndcg_at_10": ndcg_at_k(
                retrieved_papers,
                relevant_papers,
                10,
            ),
        }

        print("\nTop reranked papers:")

        for rank, result in enumerate(
            reranked_results[:5],
            start=1,
        ):
            print(
                f"{rank}. "
                f"{result['paper_id']} | "
                f"reranker={result['reranker_score']:.4f} | "
                f"{result['title']}"
            )

        print("\nMetrics:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        all_results.append(
            {
                "query_id": query_id,
                "query": query,
                "relevant_paper_ids": list(relevant_papers),
                "retrieved_paper_ids": retrieved_papers,
                "metrics": metrics,
                "results": reranked_results,
            }
        )

    # -----------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "precision_at_10",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    ]

    average_metrics = {}

    for metric_name in metric_names:
        values = [
            result["metrics"][metric_name]
            for result in all_results
        ]

        average_metrics[metric_name] = sum(values) / len(values)

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "method": "hybrid_rrf_bge_reranker",
        "query_count": len(queries),
        "dense_top_k": DENSE_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "final_top_k": FINAL_TOP_K,
        "rrf_k": RRF_K,
        "metrics": average_metrics,
        "queries": all_results,
    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("FINAL HYBRID + RERANKER RESULTS")
    print("=" * 70)

    for name, value in average_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nResults saved to:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()