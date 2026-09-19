import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "research_papers"

EMBEDDINGS_FILE = Path(
    "data/processed_clean/embeddings.json"
)

BATCH_SIZE = 100


def load_embeddings():
    print(f"Reading: {EMBEDDINGS_FILE}")

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Total embeddings: {len(data)}")

    client = QdrantClient(url=QDRANT_URL)

    total = len(data)

    for start in range(0, total, BATCH_SIZE):
        batch = data[start:start + BATCH_SIZE]

        points = []

        for index, record in enumerate(batch):
            point_id = start + index

            point = PointStruct(
                id=point_id,
                vector=record["embedding"],
                payload={
                    "chunk_id": record["chunk_id"],
                    "paper_id": record["paper_id"],
                    "title": record["title"],
                    "page_numbers": record["page_numbers"],
                    "section": record["section"],
                    "text": record["text"],
                },
            )

            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        completed = min(start + BATCH_SIZE, total)

        print(f"Uploaded {completed}/{total}")

    print()
    print("SUCCESS: All embeddings uploaded to Qdrant.")


if __name__ == "__main__":
    load_embeddings()