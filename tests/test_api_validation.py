from fastapi.testclient import TestClient

from app.api import main


class DummyGraph:
    def invoke(self, state):
        return {
            "answer": f"Answer for: {state['query']}",
            "sources": [],
        }


class FailingGraph:
    def invoke(self, state):
        raise RuntimeError("backend unavailable")


def test_health_endpoint():
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_query_is_accepted(monkeypatch):
    monkeypatch.setattr(main, "get_rag_graph", lambda: DummyGraph())

    client = TestClient(main.app)

    response = client.post(
        "/query",
        json={"query": "What is Reciprocal Rank Fusion?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "What is Reciprocal Rank Fusion?",
        "answer": "Answer for: What is Reciprocal Rank Fusion?",
        "sources": [],
    }


def test_empty_query_is_rejected():
    client = TestClient(main.app)

    response = client.post("/query", json={"query": ""})

    assert response.status_code == 422


def test_whitespace_only_query_is_rejected():
    client = TestClient(main.app)

    response = client.post("/query", json={"query": "   "})

    assert response.status_code == 422


def test_query_returns_503_when_rag_backend_is_unavailable(monkeypatch):
    monkeypatch.setattr(main, "get_rag_graph", lambda: FailingGraph())

    client = TestClient(main.app)

    response = client.post(
        "/query",
        json={"query": "What is Reciprocal Rank Fusion?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "RAG service is temporarily unavailable."
    }
