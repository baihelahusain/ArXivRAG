import requests

from app.rag.state import RAGState


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"


def rewrite_query(state: RAGState) -> RAGState:
    """
    Rewrite the user's query into a more precise
    research-paper search query using local Ollama.
    """

    query = state["query"]

    prompt = f"""
Rewrite the following user query into one precise search query
for retrieving relevant research papers.

Rules:
- Return ONLY the rewritten query.
- Do not explain your answer.
- Do not provide multiple queries.
- Do not add facts that are not present in the original query.
- Preserve the original meaning.
- Use clear academic terminology when appropriate.

Original query:
{query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    rewritten_query = data["response"].strip()

    state["rewritten_query"] = rewritten_query

    return state