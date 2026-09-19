import json
import re
from pathlib import Path


INPUT_DIR = Path("data/processed_clean")
OUTPUT_FILE = Path("data/processed_clean/chunks.json")


# Matches:
# 1 Introduction
# 2 Related Work
# 2.1 Methodology
# 3.2.1 Experimental Setup
SECTION_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$"
)


def is_section_heading(text: str) -> bool:
    """
    Check whether a line looks like a genuine numbered
    academic section heading.
    """

    text = text.strip()

    if not text:
        return False

    match = SECTION_PATTERN.match(text)

    if not match:
        return False

    section_number = match.group(1)
    heading = match.group(2).strip()

    # Reject headings that are only numeric content.
    if not re.search(r"[A-Za-z]", heading):
        return False

    # Section numbers should be reasonably short.
    if len(section_number) > 8:
        return False

    # Heading should not be excessively long.
    if len(heading) > 100:
        return False

    # Avoid sentences/table content.
    if heading.endswith((".", ",", ";", ":")):
        return False

    # A genuine heading normally contains at least one alphabetic word.
    words = heading.split()

    if len(words) == 0:
        return False

    return True


def combine_split_heading(lines: list[str], index: int):
    """
    Handles PDF extraction where the section number and title
    appear on separate lines.

    Example:
        2
        Retrieval Augmented Generation

    becomes:
        2 Retrieval Augmented Generation
    """

    if index + 1 >= len(lines):
        return None

    number = lines[index].strip()
    title = lines[index + 1].strip()

    if not re.fullmatch(r"\d+(?:\.\d+)*", number):
        return None

    if not title:
        return None

    if len(title) > 120:
        return None

    if title.endswith((".", ",", ";", ":")):
        return None

    return f"{number} {title}"


def detect_heading(lines: list[str], index: int):
    """
    Detect a section heading at the current line.

    Returns:
        heading text
        number of lines consumed
    """

    current_line = lines[index].strip()

    # Case 1:
    # 2 Retrieval Augmented Generation
    if is_section_heading(current_line):
        return current_line, 1

    # Case 2:
    # 2
    # Retrieval Augmented Generation
    combined = combine_split_heading(lines, index)

    if combined:
        return combined, 2

    # Case 3:
    # Abstract
    if current_line.lower() == "abstract":
        return "Abstract", 1

    return None, 0
def normalize_chunk_text(text: str) -> str:
    """
    Clean PDF line wrapping inside a chunk.
    """

    # Join words split across PDF line breaks.
    # Example:
    #   lan-
    #   guage
    # becomes:
    #   language
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Convert remaining line breaks into spaces.
    text = re.sub(r"\s*\n\s*", " ", text)

    # Normalize multiple spaces.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()

def create_chunks(document: dict) -> list[dict]:
    """
    Convert one parsed document into section-aware chunks.
    """

    paper_id = document["paper_id"]
    title = document.get("title", "")

    sections = []

    current_section = None
    current_text = []
    current_pages = []

    chunk_counter = 1

    def save_section():
        nonlocal current_text
        nonlocal current_pages
        nonlocal chunk_counter

        text = normalize_chunk_text("\n".join(current_text))

        if not text:
            return

        # Split large sections into smaller chunks.
        paragraphs = re.split(r"\n{2,}", text)

        chunk_text = []
        chunk_pages = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            chunk_text.append(paragraph)

            # All pages touched by this section are attached.
            for page in current_pages:
                if page not in chunk_pages:
                    chunk_pages.append(page)

            combined_text = "\n\n".join(chunk_text)

            # Approximately 1200 words per chunk.
            if len(combined_text.split()) >= 1200:
                sections.append(
                    {
                        "chunk_id": f"{paper_id}_{chunk_counter:03d}",
                        "paper_id": paper_id,
                        "title": title,
                        "page_numbers": sorted(chunk_pages),
                        "section": current_section,
                        "text": combined_text,
                    }
                )

                chunk_counter += 1
                chunk_text = []
                chunk_pages = []

        # Save remaining text.
        if chunk_text:
            sections.append(
                {
                    "chunk_id": f"{paper_id}_{chunk_counter:03d}",
                    "paper_id": paper_id,
                    "title": title,
                    "page_numbers": sorted(chunk_pages),
                    "section": current_section,
                    "text": "\n\n".join(chunk_text),
                }
            )

            chunk_counter += 1

        current_text = []
        current_pages = []

    for page in document.get("pages", []):
        page_number = page["page_number"]
        text = page.get("text", "")

        lines = text.split("\n")

        index = 0

        while index < len(lines):
            line = lines[index].strip()

            heading, consumed = detect_heading(lines, index)

            if heading:
                save_section()

                current_section = heading

                index += consumed
                continue

            if line and current_section is not None:
                current_text.append(line)

                if page_number not in current_pages:
                    current_pages.append(page_number)

            index += 1

    save_section()

    return sections


def process_all_documents():
    json_files = list(INPUT_DIR.glob("*.json"))

    # Don't process our output file as a document.
    json_files = [
        file
        for file in json_files
        if file.name != OUTPUT_FILE.name
    ]

    print(f"Found {len(json_files)} documents.")

    all_chunks = []
    failed = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                document = json.load(file)

            chunks = create_chunks(document)
            all_chunks.extend(chunks)

        except Exception as error:
            failed += 1
            print(f"Failed: {json_file.name}")
            print(f"Error: {error}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Chunking completed.")
    print(f"Documents processed: {len(json_files)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Failed documents: {failed}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_all_documents()