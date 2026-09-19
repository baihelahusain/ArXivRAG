from pathlib import Path
import json

import fitz


PDF_DIR = Path("data/raw/papers")
OUTPUT_DIR = Path("data/processed")
METADATA_FILE = Path("data/raw/metadata.json")


def load_metadata():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "page_number": page_number + 1,
            "text": text
        })

    document.close()

    return pages


def parse_pdfs():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = load_metadata()

    metadata_map = {
        paper["paper_id"]: paper
        for paper in metadata
    }

    processed = 0
    failed = 0

    for pdf_path in PDF_DIR.glob("*.pdf"):

        paper_id = pdf_path.stem

        print(f"\nProcessing: {paper_id}")

        try:

            pages = extract_text(pdf_path)

            document = {
                "paper_id": paper_id,
                "title": metadata_map.get(
                    paper_id,
                    {}
                ).get("title"),

                "authors": metadata_map.get(
                    paper_id,
                    {}
                ).get("authors", []),

                "abstract": metadata_map.get(
                    paper_id,
                    {}
                ).get("abstract"),

                "categories": metadata_map.get(
                    paper_id,
                    {}
                ).get("categories", []),

                "pages": pages
            }

            output_file = (
                OUTPUT_DIR /
                f"{paper_id}.json"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    document,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            processed += 1

            print(
                f"Saved: {output_file}"
            )

        except Exception as e:

            failed += 1

            print(
                f"FAILED: {paper_id}"
            )

            print(e)

    print("\n==============================")
    print("PDF PARSING COMPLETE")
    print("==============================")
    print(f"Processed : {processed}")
    print(f"Failed    : {failed}")
    print(f"Output    : {OUTPUT_DIR}")
    print("==============================")


if __name__ == "__main__":
    parse_pdfs()