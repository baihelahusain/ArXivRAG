import json

from sentence_transformers import CrossEncoder


QUERIES_FILE = "evaluation/queries.json"
CHUNKS_FILE = "data/processed_clean/chunks.json"

MODEL_NAME = "BAAI/bge-reranker-base"


def load_queries():
    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    queries = load_queries()
    chunks = load_chunks()

    chunk_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    model = CrossEncoder(
        MODEL_NAME,
        device="cuda",
    )

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

    print("\nEvaluating reranker score thresholds")
    print("=" * 60)

    for query_data in queries:
        query = query_data["query"]
        relevant_papers = set(query_data["relevant_paper_ids"])

        candidate_chunks = []

        for chunk in chunks:
            if chunk["paper_id"] in relevant_papers:
                candidate_chunks.append(chunk)

        if not candidate_chunks:
            continue

        pairs = [
            (query, chunk["text"])
            for chunk in candidate_chunks
        ]

        scores = model.predict(pairs)

        max_score = max(scores)

        print(f"\nQuery: {query}")
        print(f"Relevant papers: {len(relevant_papers)}")
        print(f"Maximum reranker score: {max_score:.4f}")

        for threshold in thresholds:
            passed = sum(
                1
                for score in scores
                if score >= threshold
            )

            print(
                f"  threshold {threshold:.2f}: "
                f"{passed} relevant chunks pass"
            )


if __name__ == "__main__":
    main()