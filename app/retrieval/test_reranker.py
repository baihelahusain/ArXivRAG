from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"

QUERY = "How does retrieval augmented generation reduce hallucination?"

DOCUMENTS = [
    "Retrieval-Augmented Generation reduces hallucination by retrieving relevant external information and providing it to the language model as context.",
    "This paper proposes a new graph-based method for organizing knowledge representations.",
    "The experiment compares several retrieval systems using precision and recall metrics.",
]


print("Loading reranker...")

reranker = CrossEncoder(
    MODEL_NAME,
    device="cuda"
)

print("Reranker loaded successfully")
print(f"Device: {reranker.device}")


pairs = []

for document in DOCUMENTS:
    pairs.append(
        [QUERY, document]
    )


scores = reranker.predict(pairs)


print()
print("=" * 80)
print("RERANKER TEST")
print("=" * 80)

for i, score in enumerate(scores, start=1):

    print()
    print(f"Document {i}")
    print(f"Score: {score:.4f}")
    print(f"Text: {DOCUMENTS[i - 1]}")