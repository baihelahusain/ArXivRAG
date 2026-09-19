import json

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch


# --------------------------------------------------
# Configuration
# --------------------------------------------------

QUERIES_FILE = "evaluation/queries.json"

QDRANT_COLLECTION = "research_papers"
OPENSEARCH_INDEX = "research_papers"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5


# --------------------------------------------------
# Load evaluation queries
# --------------------------------------------------

with open(QUERIES_FILE, "r", encoding="utf-8") as file:
    queries = json.load(file)

print(f"Loaded {len(queries)} evaluation queries")


# --------------------------------------------------
# Connect to Qdrant
# --------------------------------------------------

qdrant = QdrantClient(
    host="localhost",
    port=6333
)

print("Connected to Qdrant")


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
# Load embedding model
# --------------------------------------------------

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded")


# --------------------------------------------------
# Process each query
# --------------------------------------------------

for item in queries:

    query_id = item["id"]
    query = item["query"]

    print()
    print("=" * 100)
    print(f"{query_id}: {query}")
    print("=" * 100)

    # --------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    dense_results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding.tolist(),
        limit=TOP_K,
        with_payload=True
    ).points

    # --------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------

    search_body = {
        "size": TOP_K,
        "query": {
            "match": {
                "text": query
            }
        }
    }

    response = opensearch.search(
        index=OPENSEARCH_INDEX,
        body=search_body
    )

    bm25_results = response["hits"]["hits"]

    # --------------------------------------------------
    # Store candidates by paper
    # --------------------------------------------------

    candidates = {}

    # Dense candidates
    for rank, result in enumerate(dense_results, start=1):

        payload = result.payload
        paper_id = payload["paper_id"]

        if paper_id not in candidates:
            candidates[paper_id] = {
                "title": payload["title"],
                "dense_rank": rank,
                "bm25_rank": None,
                "chunk_id": payload["chunk_id"],
                "section": payload.get("section", "N/A"),
                "pages": payload.get("page_numbers", []),
                "text": payload["text"]
            }

    # BM25 candidates
    for rank, result in enumerate(bm25_results, start=1):

        source = result["_source"]
        paper_id = source["paper_id"]

        if paper_id not in candidates:

            candidates[paper_id] = {
                "title": source["title"],
                "dense_rank": None,
                "bm25_rank": rank,
                "chunk_id": source["chunk_id"],
                "section": source.get("section", "N/A"),
                "pages": source.get("page_numbers", []),
                "text": source["text"]
            }

        else:
            candidates[paper_id]["bm25_rank"] = rank

    # --------------------------------------------------
    # Display candidates
    # --------------------------------------------------

    for number, (paper_id, candidate) in enumerate(
        candidates.items(),
        start=1
    ):

        print()
        print(f"Candidate {number}")
        print(f"Paper ID: {paper_id}")
        print(f"Title: {candidate['title']}")
        print(f"Dense rank: {candidate['dense_rank']}")
        print(f"BM25 rank: {candidate['bm25_rank']}")
        print(f"Chunk ID: {candidate['chunk_id']}")
        print(f"Section: {candidate['section']}")
        print(f"Pages: {candidate['pages']}")

        text = candidate["text"].replace("\n", " ")

        print(f"Text: {text[:400]}...")