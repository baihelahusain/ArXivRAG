import json

from sentence_transformers import CrossEncoder


QUERIES_FILE = "evaluation/queries.json"
CHUNKS_FILE = "data/processed_clean/chunks.json"

MODEL_NAME = "BAAI/bge-reranker-base"


def main():
    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        queries = json.load(file)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    model = CrossEncoder(
        MODEL_NAME,
        device="cuda",
    )

    thresholds = [0.60, 0.65]

    results = {
        threshold: {
            "passed": 0,
            "failed": 0,
        }
        for threshold in thresholds
    }

    print("\nReranker threshold analysis")
    print("=" * 60)

    for query_data in queries:
        query = query_data["query"]
        relevant_papers = set(query_data["relevant_paper_ids"])

        candidate_chunks = [
            chunk
            for chunk in chunks
            if chunk["paper_id"] in relevant_papers
        ]

        pairs = [
            (query, chunk["text"])
            for chunk in candidate_chunks
        ]

        scores = model.predict(pairs)

        max_score = max(scores)

        print(f"\nQuery: {query}")
        print(f"Maximum relevant score: {max_score:.4f}")

        for threshold in thresholds:
            if max_score >= threshold:
                results[threshold]["passed"] += 1
                status = "PASS"
            else:
                results[threshold]["failed"] += 1
                status = "FAIL"

            print(f"  {threshold:.2f}: {status}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for threshold in thresholds:
        passed = results[threshold]["passed"]
        failed = results[threshold]["failed"]

        print(
            f"Threshold {threshold:.2f}: "
            f"{passed}/10 queries have a relevant chunk above threshold "
            f"({failed} below threshold)"
        )


if __name__ == "__main__":
    main()