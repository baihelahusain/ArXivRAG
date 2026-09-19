from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch


# --------------------------------------------------
# Configuration
# --------------------------------------------------

QUERY = "How does retrieval augmented generation reduce hallucination?"

TOP_K = 5
RRF_K = 60

QDRANT_COLLECTION = "research_papers"
OPENSEARCH_INDEX = "research_papers"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


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

model = SentenceTransformer(EMBEDDING_MODEL)

query_embedding = model.encode(
    QUERY,
    normalize_embeddings=True
)


# --------------------------------------------------
# Dense Search
# --------------------------------------------------

dense_results = qdrant.query_points(
    collection_name=QDRANT_COLLECTION,
    query=query_embedding.tolist(),
    limit=TOP_K,
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
    "size": TOP_K,
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
# Add Dense Results
# --------------------------------------------------

for rank, result in enumerate(dense_results, start=1):

    chunk_id = result.payload["chunk_id"]

    score = 1 / (RRF_K + rank)

    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score

    documents[chunk_id] = result.payload


# --------------------------------------------------
# Add BM25 Results
# --------------------------------------------------

for rank, result in enumerate(bm25_results, start=1):

    source = result["_source"]

    chunk_id = source["chunk_id"]

    score = 1 / (RRF_K + rank)

    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score

    documents[chunk_id] = source


# --------------------------------------------------
# Sort by RRF score
# --------------------------------------------------

ranked_results = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)


# --------------------------------------------------
# Display Results
# --------------------------------------------------

print()
print("=" * 80)
print("RRF HYBRID SEARCH RESULTS")
print("=" * 80)

for rank, (chunk_id, score) in enumerate(
    ranked_results,
    start=1
):

    document = documents[chunk_id]

    print()
    print(f"Rank {rank}")
    print(f"RRF Score: {score:.6f}")
    print(f"Chunk ID: {chunk_id}")
    print(f"Paper ID: {document['paper_id']}")
    print(f"Title: {document['title']}")
    print(f"Section: {document.get('section', 'N/A')}")
    print(f"Pages: {document.get('page_numbers', [])}")

    text = document["text"].replace("\n", " ")

    print(f"Text: {text[:300]}...")