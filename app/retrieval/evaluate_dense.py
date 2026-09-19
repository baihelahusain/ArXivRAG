import json
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

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
COLLECTION_NAME = "research_papers"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# -----------------------------
# Load evaluation queries
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

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)


# -----------------------------
# Dense retrieval function
# -----------------------------

def dense_search(query, top_k=10):
    """
    Retrieve chunks using dense vector search.
    """

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    # Convert chunk results into unique paper IDs.
    paper_ids = []

    for result in results:

        paper_id = result.payload["paper_id"]

        if paper_id not in paper_ids:
            paper_ids.append(paper_id)

    return paper_ids


# -----------------------------
# Evaluate all queries
# -----------------------------

all_results = []

print("\nStarting Dense Retrieval Evaluation...\n")


for item in queries:

    query_id = item["query_id"]
    query = item["query"]

    relevant_papers = set(item["relevant_paper_ids"])

    retrieved_papers = dense_search(
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

os.makedirs("evaluation/results", exist_ok=True)

output = {
    "method": "dense",
    "metrics": average_metrics,
    "queries": all_results,
}

with open(
    "evaluation/results/dense_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(output, file, indent=2)

print("\nDense results saved to:")
print("evaluation/results/dense_results.json")
# -----------------------------
# Display final results
# -----------------------------

print("\n" + "=" * 50)
print("DENSE RETRIEVAL RESULTS")
print("=" * 50)

for metric, value in average_metrics.items():
    print(f"{metric}: {value:.4f}")