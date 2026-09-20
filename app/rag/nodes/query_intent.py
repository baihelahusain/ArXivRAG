from langchain_ollama import ChatOllama

from app.rag.state import RAGState


OLLAMA_MODEL = "qwen3:4b"


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)


def classify_query(query: str) -> str:
    prompt = f"""
You are classifying a user query for a research-paper RAG system.

The system contains research papers mainly about:
- Retrieval-Augmented Generation
- Large Language Models
- Information Retrieval
- Embeddings
- NLP
- AI agents
- Machine Learning
- related AI research topics

User query:
{query}

Choose exactly ONE category:

research_question
The query is a specific question that can reasonably be answered
using research papers from this collection.

ambiguous
The query is too short, vague, or underspecified to determine
what research information the user wants.

outside_scope
The query is clearly unrelated to the research topics in this
system.

Rules:
- Return ONLY one category.
- Do not explain your decision.
- Do not use outside knowledge.
- Do not answer the question.

Category:
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    if "research_question" in decision:
        return "research_question"

    if "ambiguous" in decision:
        return "ambiguous"

    return "outside_scope"


def query_intent(state: RAGState) -> RAGState:
    query = state["query"]

    state["route"] = classify_query(query)

    return state