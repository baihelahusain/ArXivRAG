from langchain_ollama import ChatOllama

from app.rag.state import RAGState


OLLAMA_MODEL = "qwen3:4b"


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)


def generate_answer(state: RAGState) -> RAGState:
    query = state.get("rewritten_query", state["query"])
    documents = state.get("reranked_documents", [])

    if not documents:
        state["answer"] = (
            "I could not find enough relevant research-paper evidence "
            "to answer this question."
        )
        state["sources"] = []
        return state

    context_parts = []

    for i, document in enumerate(documents, start=1):
        context_parts.append(
            f"""
[Source {i}]
Title: {document["title"]}
Paper ID: {document["paper_id"]}
Section: {document["section"]}
Pages: {document["page_numbers"]}

Content:
{document["text"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
Answer the user's question using only the research-paper evidence below.

Question:
{query}

Evidence:
{context}

Requirements:
- Give a concise answer.
- Use only information supported by the evidence.
- Do not use outside knowledge.
- Do not invent facts or citations.
- Paraphrase the evidence.
- Cite claims using [Source 1], [Source 2], etc.
- Use only the source numbers provided.
- If the evidence is insufficient, say so.
- Do not add a references section.

Answer:
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content.strip()

    state["sources"] = [
        {
            "paper_id": document["paper_id"],
            "title": document["title"],
            "section": document["section"],
            "page_numbers": document["page_numbers"],
            "chunk_id": document["chunk_id"],
        }
        for document in documents
    ]

    return state