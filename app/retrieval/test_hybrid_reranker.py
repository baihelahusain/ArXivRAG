from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from opensearchpy import OpenSearch


# --------------------------------------------------
# Configuration
# --------------------------------------------------

QUERY = "How does retrieval augmented generation reduce hallucination?"

DENSE_TOP_K = 5
BM25_TOP_K = 5

FINAL_TOP_K = 5

RRF_K = 60

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

QDRANT_COLLECTION = "research_papers"
OPENSEARCH_INDEX = "research_papers"


# --------------------------------------------------
# Connect to Qdrant
# --------------------------------------------------

qdrant = QdrantClient(
    host="localhost",
    port=6333
)

print("Connected to Qdrant")


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cuda"
)

query_embedding = embedding_model.encode(
    QUERY,
    normalize_embeddings=True
)


# --------------------------------------------------
# Dense Search
# --------------------------------------------------

dense_results = qdrant.query_points(
    collection_name=QDRANT_COLLECTION,
    query=query_embedding.tolist(),
    limit=DENSE_TOP_K,
    with_payload=True
).points

print(f"Dense results: {len(dense_results)}")


# --------------------------------------------------
# Connect to OpenSearch
# --------------------------------------------------

opensearch = OpenSearch(
    hosts=[
        {
            "host": "localhost",
            "port": 9200
        }
    ]
)

if not opensearch.ping():
    raise RuntimeError("Could not connect to OpenSearch")

print("Connected to OpenSearch")


# --------------------------------------------------
# BM25 Search
# --------------------------------------------------

search_body = {
    "size": BM25_TOP_K,
    "query": {
        "match": {
            "text": QUERY
        }
    }
}

response = opensearch.search(
    index=OPENSEARCH_INDEX,
    body=search_body
)

bm25_results = response["hits"]["hits"]

print(f"BM25 results: {len(bm25_results)}")


# --------------------------------------------------
# RRF
# --------------------------------------------------

rrf_scores = {}
documents = {}


# --------------------------------------------------
# Add Dense Results to RRF
# --------------------------------------------------

for rank, result in enumerate(dense_results, start=1):

    chunk_id = result.payload["chunk_id"]

    rrf_score = 1 / (RRF_K + rank)

    rrf_scores[chunk_id] = (
        rrf_scores.get(chunk_id, 0)
        + rrf_score
    )

    documents[chunk_id] = result.payload


# --------------------------------------------------
# Add BM25 Results to RRF
# --------------------------------------------------

for rank, result in enumerate(bm25_results, start=1):

    source = result["_source"]

    chunk_id = source["chunk_id"]

    rrf_score = 1 / (RRF_K + rank)

    rrf_scores[chunk_id] = (
        rrf_scores.get(chunk_id, 0)
        + rrf_score
    )

    documents[chunk_id] = source


# --------------------------------------------------
# Sort RRF results
# --------------------------------------------------

rrf_results = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)


# --------------------------------------------------
# Take RRF candidates
# --------------------------------------------------

candidates = []

for chunk_id, rrf_score in rrf_results:

    document = documents[chunk_id]

    candidates.append(
        {
            "chunk_id": chunk_id,
            "rrf_score": rrf_score,
            "document": document
        }
    )


# --------------------------------------------------
# Load Reranker
# --------------------------------------------------

print()
print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL,
    device="cuda"
)

print("Reranker loaded successfully")
print(f"Reranker device: {reranker.device}")


# --------------------------------------------------
# Prepare Query + Document pairs
# --------------------------------------------------

pairs = []

for candidate in candidates:

    document_text = candidate["document"]["text"]

    pairs.append(
        [
            QUERY,
            document_text
        ]
    )


# --------------------------------------------------
# Rerank
# --------------------------------------------------

reranker_scores = reranker.predict(
    pairs
)


# --------------------------------------------------
# Attach reranker scores
# --------------------------------------------------

for candidate, score in zip(
    candidates,
    reranker_scores
):

    candidate["reranker_score"] = float(score)


# --------------------------------------------------
# Sort by reranker score
# --------------------------------------------------

final_results = sorted(
    candidates,
    key=lambda item: item["reranker_score"],
    reverse=True
)


# --------------------------------------------------
# Display Final Results
# --------------------------------------------------

print()
print("=" * 80)
print("HYBRID + RERANKER RESULTS")
print("=" * 80)

for rank, result in enumerate(
    final_results[:FINAL_TOP_K],
    start=1
):

    document = result["document"]

    print()
    print(f"Rank {rank}")
    print(f"Reranker Score: {result['reranker_score']:.4f}")
    print(f"RRF Score: {result['rrf_score']:.6f}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Paper ID: {document['paper_id']}")
    print(f"Title: {document['title']}")
    print(f"Section: {document.get('section', 'N/A')}")
    print(f"Pages: {document.get('page_numbers', [])}")

    text = document["text"].replace("\n", " ")

    print(f"Text: {text[:500]}...")