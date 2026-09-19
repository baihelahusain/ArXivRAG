from pathlib import Path
import json

from opensearchpy import OpenSearch, helpers


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200

INDEX_NAME = "research_papers"

CHUNKS_FILE = Path("data/processed_clean/chunks.json")


# ---------------------------------------------------------
# Connect to OpenSearch
# ---------------------------------------------------------

client = OpenSearch(
    hosts=[
        {
            "host": OPENSEARCH_HOST,
            "port": OPENSEARCH_PORT,
        }
    ],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False,
)


# ---------------------------------------------------------
# Check connection
# ---------------------------------------------------------

if not client.ping():
    raise RuntimeError("Could not connect to OpenSearch.")

print("Connected to OpenSearch")


# ---------------------------------------------------------
# Delete existing index if it exists
# ---------------------------------------------------------

if client.indices.exists(index=INDEX_NAME):
    print(f"Index '{INDEX_NAME}' already exists.")
    print("Deleting existing index...")
    client.indices.delete(index=INDEX_NAME)


# ---------------------------------------------------------
# Create index
# ---------------------------------------------------------

index_body = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {
                "type": "keyword"
            },
            "paper_id": {
                "type": "keyword"
            },
            "title": {
                "type": "text"
            },
            "section": {
                "type": "text"
            },
            "page_numbers": {
                "type": "integer"
            },
            "text": {
                "type": "text"
            }
        }
    }
}


client.indices.create(
    index=INDEX_NAME,
    body=index_body
)

print(f"Created index: {INDEX_NAME}")


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Chunks file not found: {CHUNKS_FILE}"
    )

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


# ---------------------------------------------------------
# Prepare bulk documents
# ---------------------------------------------------------

def generate_documents():
    for chunk in chunks:
        yield {
            "_index": INDEX_NAME,
            "_source": {
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "page_numbers": chunk["page_numbers"],
                "section": chunk["section"],
                "text": chunk["text"],
            },
        }


# ---------------------------------------------------------
# Bulk index
# ---------------------------------------------------------

success, failed = helpers.bulk(
    client,
    generate_documents(),
    chunk_size=500,
    request_timeout=120,
    raise_on_error=False,
)

print(f"Successfully indexed: {success}")
print(f"Failed documents: {len(failed)}")


# ---------------------------------------------------------
# Refresh index
# ---------------------------------------------------------

client.indices.refresh(index=INDEX_NAME)


# ---------------------------------------------------------
# Verify document count
# ---------------------------------------------------------

count_response = client.count(index=INDEX_NAME)

print(
    f"Documents in '{INDEX_NAME}': "
    f"{count_response['count']}"
)

print("BM25 index setup completed.")