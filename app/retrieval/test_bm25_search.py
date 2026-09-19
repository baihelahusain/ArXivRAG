from opensearchpy import OpenSearch


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
INDEX_NAME = "research_papers"

QUERY = "How does retrieval augmented generation reduce hallucination?"

TOP_K = 5


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


if not client.ping():
    raise RuntimeError("Could not connect to OpenSearch.")

print("Connected to OpenSearch")


# ---------------------------------------------------------
# BM25 search
# ---------------------------------------------------------

search_body = {
    "size": TOP_K,
    "query": {
        "match": {
            "text": {
                "query": QUERY
            }
        }
    }
}


response = client.search(
    index=INDEX_NAME,
    body=search_body
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

results = response["hits"]["hits"]

print()
print(f"Query: {QUERY}")
print(f"Results: {len(results)}")
print("=" * 80)


for rank, result in enumerate(results, start=1):

    source = result["_source"]

    print(f"\nRank: {rank}")
    print(f"BM25 Score: {result['_score']:.4f}")
    print(f"Chunk ID: {source['chunk_id']}")
    print(f"Paper ID: {source['paper_id']}")
    print(f"Title: {source['title']}")
    print(f"Section: {source['section']}")
    print(f"Pages: {source['page_numbers']}")

    preview = source["text"].replace("\n", " ")

    if len(preview) > 400:
        preview = preview[:400] + "..."

    print(f"Text: {preview}")