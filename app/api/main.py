from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from app.rag.graph import rag_graph


app = FastAPI(
    title="Research Paper RAG API",
    version="1.0.0",
)

@app.get('/')
def welcome():
    return{'message':"Welcom to ArXiv RAG"}


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Question to ask the research-paper RAG system",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Query cannot be empty or whitespace-only.")

        return value
    
class Source(BaseModel):
    paper_id: str
    title: str
    section: str
    page_numbers: list[int]
    chunk_id: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        result = rag_graph.invoke({
            "query": request.query
        })

        return QueryResponse(
            query=request.query,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
        )

    except Exception as error:
        print(f"RAG pipeline error: {error}")

        raise HTTPException(
            status_code=503,
            detail="RAG service is temporarily unavailable.",
        )