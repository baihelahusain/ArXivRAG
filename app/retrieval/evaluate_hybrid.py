import json
import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from opensearchpy import OpenSearch

from app.retrieval.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
)


# -----------------------------
# Configuration
# -----------------------------

QUERIES_FILE = "evaluation/queries.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "research_papers"

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_INDEX = "research_papers"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

RRF_K = 60


# -----------------------------
# Load queries
# -----------------------------

with open(QUERIES_FILE, "r", encoding="utf-8") as file:
    queries = json.load(file)


# -----------------------------
# Load embedding model
# -----------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cuda",
)

print("Embedding model loaded.")


# -----------------------------
# Connect to Qdrant
# -----------------------------

qdrant = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)


# -----------------------------
# Connect to OpenSearch
# -----------------------------

opensearch = OpenSearch(
    hosts=[
        {
            "host": OPENSEARCH_HOST,
            "port": OPENSEARCH_PORT,
        }
    ]
)

print("Connected to OpenSearch.")


# -----------------------------
# Dense Search
# -----------------------------

def dense_search(query, top_k=10):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    paper_ids = []

    for result in results:

        paper_id = result.payload["paper_id"]

        if paper_id not in paper_ids:
            paper_ids.append(paper_id)

    return paper_ids


# -----------------------------
# BM25 Search
# -----------------------------

def bm25_search(query, top_k=10):

    response = opensearch.search(
        index=OPENSEARCH_INDEX,
        body={
            "size": top_k,
            "query": {
                "match": {
                    "text": {
                        "query": query
                    }
                }
            }
        }
    )

    paper_ids = []

    for hit in response["hits"]["hits"]:

        paper_id = hit["_source"]["paper_id"]

        if paper_id not in paper_ids:
            paper_ids.append(paper_id)

    return paper_ids


# -----------------------------
# Reciprocal Rank Fusion
# -----------------------------

def rrf_fusion(dense_results, bm25_results):

    scores = {}

    # Dense ranking
    for rank, paper_id in enumerate(
        dense_results,
        start=1,
    ):

        score = 1 / (RRF_K + rank)

        scores[paper_id] = (
            scores.get(paper_id, 0) + score
        )

    # BM25 ranking
    for rank, paper_id in enumerate(
        bm25_results,
        start=1,
    ):

        score = 1 / (RRF_K + rank)

        scores[paper_id] = (
            scores.get(paper_id, 0) + score
        )

    # Sort by RRF score
    ranked_papers = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        paper_id
        for paper_id, score in ranked_papers
    ]


# -----------------------------
# Evaluate queries
# -----------------------------

all_results = []

print("\nStarting Hybrid Retrieval Evaluation...\n")


for item in queries:

    query_id = item["query_id"]
    query = item["query"]

    relevant_papers = set(
        item["relevant_paper_ids"]
    )

    dense_results = dense_search(
        query,
        top_k=10,
    )

    bm25_results = bm25_search(
        query,
        top_k=10,
    )

    hybrid_results = rrf_fusion(
        dense_results,
        bm25_results,
    )

    result = {
        "query_id": query_id,
        "query": query,

        "dense_results": dense_results,

        "bm25_results": bm25_results,

        "hybrid_results": hybrid_results,

        "relevant_papers": list(
            relevant_papers
        ),

        "precision_at_5": precision_at_k(
            hybrid_results,
            relevant_papers,
            5,
        ),

        "recall_at_5": recall_at_k(
            hybrid_results,
            relevant_papers,
            5,
        ),

        "precision_at_10": precision_at_k(
            hybrid_results,
            relevant_papers,
            10,
        ),

        "recall_at_10": recall_at_k(
            hybrid_results,
            relevant_papers,
            10,
        ),

        "mrr": mrr(
            hybrid_results,
            relevant_papers,
        ),

        "ndcg_at_5": ndcg_at_k(
            hybrid_results,
            relevant_papers,
            5,
        ),

        "ndcg_at_10": ndcg_at_k(
            hybrid_results,
            relevant_papers,
            10,
        ),
    }

    all_results.append(result)

    print(
        f"{query_id}: "
        f"Recall@5={result['recall_at_5']:.3f}, "
        f"Recall@10={result['recall_at_10']:.3f}, "
        f"MRR={result['mrr']:.3f}"
    )


# -----------------------------
# Average metrics
# -----------------------------

num_queries = len(all_results)

average_metrics = {
    "precision_at_5": sum(
        r["precision_at_5"]
        for r in all_results
    ) / num_queries,

    "recall_at_5": sum(
        r["recall_at_5"]
        for r in all_results
    ) / num_queries,

    "precision_at_10": sum(
        r["precision_at_10"]
        for r in all_results
    ) / num_queries,

    "recall_at_10": sum(
        r["recall_at_10"]
        for r in all_results
    ) / num_queries,

    "mrr": sum(
        r["mrr"]
        for r in all_results
    ) / num_queries,

    "ndcg_at_5": sum(
        r["ndcg_at_5"]
        for r in all_results
    ) / num_queries,

    "ndcg_at_10": sum(
        r["ndcg_at_10"]
        for r in all_results
    ) / num_queries,
}


# -----------------------------
# Save results
# -----------------------------

os.makedirs(
    "evaluation/results",
    exist_ok=True,
)

output = {
    "method": "hybrid_rrf",
    "rrf_k": RRF_K,
    "metrics": average_metrics,
    "queries": all_results,
}

with open(
    "evaluation/results/hybrid_results.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
    )


# -----------------------------
# Display results
# -----------------------------

print("\n" + "=" * 50)
print("HYBRID RRF RESULTS")
print("=" * 50)

for metric, value in average_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nHybrid results saved to:")
print("evaluation/results/hybrid_results.json")