from langchain_ollama import ChatOllama

from app.rag.state import RAGState


OLLAMA_MODEL = "qwen3:4b"


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)


def grade_document(query: str, document: dict) -> str:
    prompt = f"""
You are evaluating whether a retrieved research-paper passage
is useful for answering a user's question.

User question:
{query}

Research-paper passage:
{document["text"]}

Decide whether the passage contains information that directly
helps answer the user's question.

Rules:
- Return ONLY one word.
- Return "relevant" if the passage provides useful evidence
  for answering the question.
- Return "not_relevant" if it does not.
- Do not use outside knowledge.
- Do not explain your decision.

Decision:
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    if "relevant" in decision and "not_relevant" not in decision:
        return "relevant"

    return "not_relevant"


def llm_grade_documents(state: RAGState) -> RAGState:
    query = state.get("rewritten_query", state["query"])
    documents = state.get("reranked_documents", [])

    graded_documents = []

    for document in documents:
        grade = grade_document(query, document)

        graded_document = document.copy()
        graded_document["llm_relevance"] = grade

        graded_documents.append(graded_document)

    state["reranked_documents"] = graded_documents

    relevant_count = sum(
        1
        for document in graded_documents
        if document["llm_relevance"] == "relevant"
    )

    if relevant_count > 0:
        state["relevance_grade"] = "relevant"
    else:
        state["relevance_grade"] = "not_relevant"

    return state