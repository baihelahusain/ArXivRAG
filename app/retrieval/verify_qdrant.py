from qdrant_client import QdrantClient


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "research_papers"


def verify_collection():
    client = QdrantClient(url=QDRANT_URL)

    collection = client.get_collection(COLLECTION_NAME)

    print("Qdrant collection verification")
    print("------------------------------")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points: {collection.points_count}")
    print(f"Status: {collection.status}")

    vectors = collection.config.params.vectors

    print(f"Vector size: {vectors.size}")
    print(f"Distance: {vectors.distance}")

    if vectors.size == 384 and str(vectors.distance) == "Cosine":
        print()
        print("SUCCESS: Qdrant collection is configured correctly.")
    else:
        print()
        print("WARNING: Collection configuration is unexpected.")


if __name__ == "__main__":
    verify_collection()
    