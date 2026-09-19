from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "research_papers"

TOP_K = 5


def search(query: str):
    print("Loading embedding model...")
    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device="cuda"
    )

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    print("Creating query embedding...")
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    print("\nSearching...\n")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    )

    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    for rank, point in enumerate(results.points, start=1):
        payload = point.payload

        print(f"\nResult {rank}")
        print("-" * 80)
        print(f"Score   : {point.score:.4f}")
        print(f"Paper   : {payload['title']}")
        print(f"Section : {payload['section']}")
        print(f"Pages   : {payload['page_numbers']}")
        print(f"Chunk ID: {payload['chunk_id']}")
        print("\nPreview:")
        print(payload["text"][:500])
        print()


if __name__ == "__main__":
    query = "How does retrieval augmented generation reduce hallucination?"
    search(query)