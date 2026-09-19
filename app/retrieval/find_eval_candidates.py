import json

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

QUERIES_FILE = "evaluation/queries.json"

QDRANT_COLLECTION = "research_papers"

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
# Load embedding model
# --------------------------------------------------

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded")


# --------------------------------------------------
# Search each evaluation query
# --------------------------------------------------

for item in queries:

    query_id = item["id"]
    query = item["query"]

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding.tolist(),
        limit=TOP_K,
        with_payload=True
    ).points

    print()
    print("=" * 90)
    print(f"{query_id}: {query}")
    print("=" * 90)

    for rank, result in enumerate(results, start=1):

        payload = result.payload

        print()
        print(f"Rank {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Paper ID: {payload['paper_id']}")
        print(f"Chunk ID: {payload['chunk_id']}")
        print(f"Title: {payload['title']}")
        print(f"Section: {payload.get('section', 'N/A')}")
        print(f"Pages: {payload.get('page_numbers', [])}")

        text = payload["text"].replace("\n", " ")

        print(f"Text: {text[:250]}...")