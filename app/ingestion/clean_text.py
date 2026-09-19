import json
import re
from pathlib import Path


INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed_clean")


def fix_encoding(text: str) -> str:
    """
    Fix common UTF-8 decoding artifacts found in extracted PDF text.
    """

    replacements = {
        "â€™": "’",
        "â€œ": "“",
        "â€": "”",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "âˆ—": "∗",
        "Â©": "©",
        "Â®": "®",
        "Â": "",
    }

    for bad, correct in replacements.items():
        text = text.replace(bad, correct)

    return text

def fix_hyphenation(text: str) -> str:
    """
    Join words that were split across PDF line breaks.

    Example:
        lan-
        guage

    becomes:
        language
    """

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text

def normalize_text(text: str) -> str:
    text = fix_encoding(text)
    text = fix_hyphenation(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces from each line.
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Convert line breaks into spaces.
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)

    # Normalize multiple spaces.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def clean_document(document: dict) -> dict:
    """
    Clean all page text while preserving document structure and metadata.
    """

    cleaned_document = document.copy()

    cleaned_pages = []

    for page in document.get("pages", []):
        cleaned_page = page.copy()

        original_text = page.get("text", "")
        cleaned_text = normalize_text(original_text)

        cleaned_page["text"] = cleaned_text

        cleaned_pages.append(cleaned_page)

    cleaned_document["pages"] = cleaned_pages

    return cleaned_document


def clean_all_documents() -> None:
    """
    Clean every parsed JSON document.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(INPUT_DIR.glob("*.json"))

    print(f"Found {len(json_files)} JSON documents.")

    success = 0
    failed = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                document = json.load(file)

            cleaned_document = clean_document(document)

            output_file = OUTPUT_DIR / json_file.name

            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(
                    cleaned_document,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            success += 1

        except Exception as error:
            failed += 1
            print(f"Failed: {json_file.name}")
            print(f"Error: {error}")

    print("\nCleaning completed.")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    clean_all_documents()