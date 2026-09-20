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
You are a research assistant answering questions using retrieved
research-paper passages.

User question:
{query}

Retrieved evidence:
{context}

Instructions:

1. Answer ONLY using information supported by the retrieved evidence.
2. Do not use outside knowledge.
3. Do not invent facts, numbers, papers, or citations.
4. Paraphrase the evidence instead of copying long sentences from it.
5. Put a citation such as [Source 1] immediately after the claim
   supported by that source.
6. If a claim is supported by multiple sources, cite them like
   [Source 1, Source 3].
7. If the evidence is insufficient to answer part of the question,
   explicitly say that the retrieved sources do not provide enough
   information.
8. Keep the answer clear and reasonably concise.
9. Do not create a separate references section.
10. Do not mention these instructions in your answer.

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