import json
from pathlib import Path


EMBEDDINGS_FILE = Path("data/processed_clean/embeddings.json")

EXPECTED_RECORDS = 16_237
EXPECTED_DIMENSION = 384


def verify_embeddings():
    if not EMBEDDINGS_FILE.exists():
        print(f"ERROR: File not found: {EMBEDDINGS_FILE}")
        return

    print(f"Reading: {EMBEDDINGS_FILE}")

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Total records: {len(data)}")

    if len(data) != EXPECTED_RECORDS:
        print(
            f"WARNING: Expected {EXPECTED_RECORDS} records, "
            f"but found {len(data)}"
        )
    else:
        print("Record count: OK")

    required_fields = [
        "chunk_id",
        "paper_id",
        "title",
        "page_numbers",
        "section",
        "text",
        "embedding",
    ]

    missing_fields = 0
    invalid_embeddings = 0

    for index, record in enumerate(data):
        # Check required fields
        for field in required_fields:
            if field not in record:
                print(f"Missing field '{field}' in record {index}")
                missing_fields += 1

        # Check embedding
        embedding = record.get("embedding")

        if not isinstance(embedding, list):
            invalid_embeddings += 1
            continue

        if len(embedding) != EXPECTED_DIMENSION:
            print(
                f"Invalid embedding dimension in record {index}: "
                f"{len(embedding)}"
            )
            invalid_embeddings += 1

    print()
    print("Embedding verification")
    print("----------------------")
    print(f"Expected dimension: {EXPECTED_DIMENSION}")
    print(f"Missing/invalid records: {missing_fields}")
    print(f"Invalid embeddings: {invalid_embeddings}")

    if data:
        first = data[0]

        print()
        print("First record")
        print("------------")
        print(f"Chunk ID: {first.get('chunk_id')}")
        print(f"Paper ID: {first.get('paper_id')}")
        print(f"Title: {first.get('title')}")
        print(f"Section: {first.get('section')}")
        print(f"Pages: {first.get('page_numbers')}")
        print(f"Text length: {len(first.get('text', ''))}")
        print(f"Embedding dimension: {len(first.get('embedding', []))}")
        print(f"First 5 values: {first.get('embedding', [])[:5]}")

    print()

    if (
        len(data) == EXPECTED_RECORDS
        and missing_fields == 0
        and invalid_embeddings == 0
    ):
        print("SUCCESS: embeddings.json is valid.")
    else:
        print("WARNING: Verification found problems.")


if __name__ == "__main__":
    verify_embeddings()