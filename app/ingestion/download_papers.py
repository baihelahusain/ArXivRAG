import json
import time
from pathlib import Path
from urllib.request import urlretrieve

import arxiv


NUM_PAPERS = 500

SEARCH_QUERIES = [
    '"retrieval augmented generation"',
    '"large language models"',
    '"information retrieval"',
    '"dense retrieval"',
    '"vector database"',
    '"question answering"',
    '"LLM agents"',
]

OUTPUT_DIR = Path("data/raw/papers")
METADATA_FILE = Path("data/raw/metadata.json")


def download_papers():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=3,
    )

    downloaded = []
    seen_ids = set()

    for query_text in SEARCH_QUERIES:

        print(f"\nSearching: {query_text}")

        search = arxiv.Search(
            query=query_text,
            max_results=NUM_PAPERS,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in client.results(search):

            paper_id = result.entry_id.split("/")[-1]

            if paper_id in seen_ids:
                continue

            seen_ids.add(paper_id)

            pdf_path = OUTPUT_DIR / f"{paper_id}.pdf"

            if pdf_path.exists():
                print(f"Already exists: {paper_id}")
                continue

            print(f"Downloading: {paper_id}")
            print(f"Title: {result.title}")

            try:
                # Current arxiv package:
                # download using the PDF URL
                urlretrieve(
                    result.pdf_url,
                    str(pdf_path),
                )

                downloaded.append(
                    {
                        "paper_id": paper_id,
                        "title": result.title,
                        "authors": [
                            author.name for author in result.authors
                        ],
                        "abstract": result.summary,
                        "categories": result.categories,
                        "published": result.published.isoformat(),
                        "updated": result.updated.isoformat(),
                        "pdf_url": result.pdf_url,
                    }
                )

                print(f"Saved: {pdf_path}")

            except Exception as e:
                print(f"Failed: {paper_id}")
                print(f"Error: {e}")

            # Respect arXiv's rate limits
            time.sleep(3)

            if len(seen_ids) >= NUM_PAPERS:
                break

        if len(seen_ids) >= NUM_PAPERS:
            break

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(
            downloaded,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n==============================")
    print(f"Downloaded papers: {len(downloaded)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata: {METADATA_FILE}")
    print("==============================")


if __name__ == "__main__":
    download_papers()