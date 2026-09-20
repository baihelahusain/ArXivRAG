import re


def is_reference_only(document: dict) -> bool:
    """
    Return True only when a chunk strongly looks like
    bibliography/reference material rather than research content.
    """

    text = document.get("text", "").strip()

    if not text:
        return False

    # Strong signals that the chunk is a bibliography.
    reference_markers = len(
        re.findall(r"\[\d+\]", text)
    )

    url_markers = len(
        re.findall(r"(https?://|doi:)", text, re.IGNORECASE)
    )

    # Detect consecutive numbered references such as:
    # [20] ...
    # [21] ...
    # [22] ...
    numbered_references = len(
        re.findall(r"\[\d+\]\s+[A-Z]", text)
    )

    # Conservative rule:
    # Require several reference-like signals before filtering.
    if reference_markers >= 5 and url_markers >= 2:
        return True

    if numbered_references >= 4 and url_markers >= 2:
        return True

    return False