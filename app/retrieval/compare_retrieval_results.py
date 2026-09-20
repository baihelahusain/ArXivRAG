import json
import csv
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "evaluation" / "results"

DENSE_FILE = RESULTS_DIR / "dense_results.json"
BM25_FILE = RESULTS_DIR / "bm25_results.json"
HYBRID_FILE = RESULTS_DIR / "hybrid_results.json"
RERANKER_FILE = RESULTS_DIR / "reranker_results.json"

OUTPUT_FILE = RESULTS_DIR / "retrieval_comparison.csv"


# ---------------------------------------------------------
# Load result file
# ---------------------------------------------------------

def load_metrics(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["metrics"]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading evaluation results...")

    results = {
        "Dense": load_metrics(DENSE_FILE),
        "BM25": load_metrics(BM25_FILE),
        "Hybrid RRF": load_metrics(HYBRID_FILE),
        "Hybrid + Reranker": load_metrics(RERANKER_FILE),
    }

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "precision_at_10",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    ]

    # -----------------------------------------------------
    # Print comparison
    # -----------------------------------------------------

    print("\n" + "=" * 90)
    print("RETRIEVAL ABLATION COMPARISON")
    print("=" * 90)

    header = (
        f"{'Metric':<20}"
        f"{'Dense':>15}"
        f"{'BM25':>15}"
        f"{'Hybrid RRF':>15}"
        f"{'Hybrid + Reranker':>20}"
    )

    print(header)
    print("-" * 90)

    for metric in metric_names:

        row = (
            f"{metric:<20}"
            f"{results['Dense'][metric]:>15.4f}"
            f"{results['BM25'][metric]:>15.4f}"
            f"{results['Hybrid RRF'][metric]:>15.4f}"
            f"{results['Hybrid + Reranker'][metric]:>20.4f}"
        )

        print(row)

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "metric",
                "dense",
                "bm25",
                "hybrid_rrf",
                "hybrid_reranker",
            ]
        )

        for metric in metric_names:

            writer.writerow(
                [
                    metric,
                    results["Dense"][metric],
                    results["BM25"][metric],
                    results["Hybrid RRF"][metric],
                    results["Hybrid + Reranker"][metric],
                ]
            )

    print("\nComparison saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()