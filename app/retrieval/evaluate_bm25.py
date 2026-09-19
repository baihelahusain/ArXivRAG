import json
import os

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

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200

INDEX_NAME = "research_papers"


# -----------------------------
# Load evaluation queries
# -----------------------------

with open(QUERIES_FILE, "r", encoding="utf-8") as file:
    queries = json.load(file)


# -----------------------------
# Connect to OpenSearch
# -----------------------------

client = OpenSearch(
    hosts=[
        {
            "host": OPENSEARCH_HOST,
            "port": OPENSEARCH_PORT,
        }
    ]
)

print("Connected to OpenSearch.")


# -----------------------------
# BM25 retrieval function
# -----------------------------

def bm25_search(query, top_k=10):
    """
    Retrieve papers using BM25 keyword search.
    """

    response = client.search(
        index=INDEX_NAME,
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

        # Avoid counting multiple chunks
        # from the same paper more than once.
        if paper_id not in paper_ids:
            paper_ids.append(paper_id)

    return paper_ids


# -----------------------------
# Evaluate all queries
# -----------------------------

all_results = []

print("\nStarting BM25 Retrieval Evaluation...\n")


for item in queries:

    query_id = item["query_id"]
    query = item["query"]

    relevant_papers = set(
        item["relevant_paper_ids"]
    )

    retrieved_papers = bm25_search(
        query,
        top_k=10,
    )

    result = {
        "query_id": query_id,
        "query": query,
        "retrieved_papers": retrieved_papers,
        "relevant_papers": list(relevant_papers),

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

    all_results.append(result)

    print(
        f"{query_id}: "
        f"Recall@5={result['recall_at_5']:.3f}, "
        f"Recall@10={result['recall_at_10']:.3f}, "
        f"MRR={result['mrr']:.3f}"
    )


# -----------------------------
# Calculate average metrics
# -----------------------------

num_queries = len(all_results)

average_metrics = {
    "precision_at_5": sum(
        r["precision_at_5"] for r in all_results
    ) / num_queries,

    "recall_at_5": sum(
        r["recall_at_5"] for r in all_results
    ) / num_queries,

    "precision_at_10": sum(
        r["precision_at_10"] for r in all_results
    ) / num_queries,

    "recall_at_10": sum(
        r["recall_at_10"] for r in all_results
    ) / num_queries,

    "mrr": sum(
        r["mrr"] for r in all_results
    ) / num_queries,

    "ndcg_at_5": sum(
        r["ndcg_at_5"] for r in all_results
    ) / num_queries,

    "ndcg_at_10": sum(
        r["ndcg_at_10"] for r in all_results
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
    "method": "bm25",
    "metrics": average_metrics,
    "queries": all_results,
}

with open(
    "evaluation/results/bm25_results.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
    )


# -----------------------------
# Display final results
# -----------------------------

print("\n" + "=" * 50)
print("BM25 RETRIEVAL RESULTS")
print("=" * 50)

for metric, value in average_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nBM25 results saved to:")
print("evaluation/results/bm25_results.json")