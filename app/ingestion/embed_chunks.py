import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed_clean/chunks.json")
OUTPUT_FILE = Path("data/processed_clean/embeddings.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")

    start_time = time.perf_counter()

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model_load_time = time.perf_counter() - start_time

    print(f"Model loaded in: {model_load_time:.2f} seconds")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model device: {model.device}")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")
    print(f"Batch size: {BATCH_SIZE}")

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_time = time.perf_counter() - start_time

    print("\nEmbedding generation completed.")

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Total embedding time: {embedding_time:.2f} seconds")

    if embedding_time > 0:
        print(
            f"Average time per chunk: "
            f"{embedding_time / len(chunks):.4f} seconds"
        )

    output_data = []

    for chunk, embedding in zip(chunks, embeddings):
        output_data.append(
            {
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "page_numbers": chunk["page_numbers"],
                "section": chunk["section"],
                "text": chunk["text"],
                "embedding": embedding.tolist(),
            }
        )

    print("\nSaving embeddings...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output_data, file)

    print(f"Saved embeddings to: {OUTPUT_FILE}")

    print("\nEmbedding pipeline completed successfully.")


if __name__ == "__main__":
    main()