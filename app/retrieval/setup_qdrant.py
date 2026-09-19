from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "research_papers"


def create_collection():
    client = QdrantClient(url=QDRANT_URL)

    collections = client.get_collections().collections
    existing_names = [collection.name for collection in collections]

    if COLLECTION_NAME in existing_names:
        print(f"Collection already exists: {COLLECTION_NAME}")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    print(f"Collection created: {COLLECTION_NAME}")


if __name__ == "__main__":
    create_collection()