from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"

queries = [
    "What is the capital of France?",
    "How do I cook biryani?",
    "What is today's weather?",
    "Who won the cricket match?",
    "How do I repair a bicycle?",
]


def main():
    model = CrossEncoder(
        MODEL_NAME,
        device="cuda",
    )

    documents = [
        {
            "title": "Retrieval-Augmented Generation",
            "text": (
                "Retrieval-Augmented Generation retrieves external "
                "information to improve language model responses."
            ),
        },
        {
            "title": "Information Retrieval",
            "text": (
                "BM25 and dense retrieval are commonly used for "
                "retrieving relevant documents."
            ),
        },
        {
            "title": "Large Language Models",
            "text": (
                "Large language models can use retrieved context "
                "to improve factuality and reduce hallucination."
            ),
        },
    ]

    print("\nIrrelevant query threshold test")
    print("=" * 70)

    for query in queries:
        pairs = [
            (query, document["text"])
            for document in documents
        ]

        scores = model.predict(pairs)
        max_score = max(scores)

        print(f"\nQuery: {query}")
        print(f"Maximum reranker score: {max_score:.4f}")

        if max_score >= 0.65:
            print("Result: ABOVE 0.65")
        else:
            print("Result: BELOW 0.65")


if __name__ == "__main__":
    main()