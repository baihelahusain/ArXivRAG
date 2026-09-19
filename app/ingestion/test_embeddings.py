import json
import time
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL_NAME)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Model device: {model.device}")
CHUNKS_FILE = Path("data/processed_clean/chunks.json")



def load_test_chunks(limit: int = 10) -> list[dict]:
    """Load a small number of chunks for testing."""

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks[:limit]


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")

    start_time = time.perf_counter()

    model = SentenceTransformer(MODEL_NAME)

    model_load_time = time.perf_counter() - start_time

    print(f"Model loaded in: {model_load_time:.2f} seconds")

    chunks = load_test_chunks(10)

    print(f"Loaded {len(chunks)} chunks")

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embedding_time = time.perf_counter() - start_time

    print("\nEmbedding test completed.")

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embedding generation time: {embedding_time:.2f} seconds")

    if embedding_time > 0:
        print(
            f"Average time per chunk: "
            f"{embedding_time / len(texts):.4f} seconds"
        )

    print("\nFirst embedding:")
    print(embeddings[0][:10])

    print("\nFirst chunk:")
    print(texts[0][:300])

    print("\nTest successful.")


if __name__ == "__main__":
    main()