from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Production RAG API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query():
    response = client.post(
        "/query",
        json={
            "question": "What is RAG?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0


def test_query_validation():
    response = client.post(
        "/query",
        json={
            "question": "a"
        },
    )

    assert response.status_code == 422
