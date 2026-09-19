from pathlib import Path
from urllib.request import urlretrieve
import json
import time


PAPERS_DIR = Path("data/raw/papers")
METADATA_FILE = Path("data/raw/metadata.json")

PAPER_IDS = [
    "2512.20626v2",
    "2605.24366v1",
]


def redownload():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata_map = {
        paper["paper_id"]: paper
        for paper in metadata
    }

    for paper_id in PAPER_IDS:

        paper = metadata_map.get(paper_id)

        if not paper:
            print(f"Metadata not found: {paper_id}")
            continue

        url = paper["pdf_url"]
        output = PAPERS_DIR / f"{paper_id}.pdf"

        print(f"\nDownloading: {paper_id}")
        print(f"URL: {url}")

        try:
            urlretrieve(url, output)

            size = output.stat().st_size

            print(f"Saved: {output}")
            print(f"Size: {size:,} bytes")

        except Exception as e:
            print(f"FAILED: {paper_id}")
            print(e)

        time.sleep(3)


if __name__ == "__main__":
    redownload()