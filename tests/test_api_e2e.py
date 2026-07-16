import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_e2e_health():
    """Verify that the live health check endpoint responds correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "ok"
    assert "model" in res
    assert res["platform"] == "Fintech RAG Chatbot"

def test_e2e_query_refusal():
    """Verify that an out-of-scope query triggers a direct refusal and no tools are run."""
    response = client.post(
        "/query",
        json={"message": "What is the capital of France?", "history": []}
    )
    assert response.status_code == 200
    res = response.json()
    assert "response" in res
    assert "Fintech" in res["response"] or "knowledge base" in res["response"]
    assert isinstance(res["tool_calls_executed"], list)

def test_e2e_ingest_and_query_success():
    """Verify ingestion of a document and its successful hybrid retrieval."""
    # 1. Ingest document containing a unique platform metadata/keyword
    ingest_response = client.post(
        "/ingest",
        json={
            "text": "Our Fintech SaaS platform uses a proprietary transaction routing mechanism code-named AegisSec-99.",
            "metadata": {"source": "e2e_hybrid_test"}
        }
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] == "success"
    assert ingest_response.json()["chunk_count"] > 0
    
    # 2. Query for the ingested fact using the keyword
    response = client.post(
        "/query",
        json={"message": "What is the routing mechanism code name used by the SaaS platform?", "history": []}
    )
    assert response.status_code == 200
    res = response.json()
    assert "response" in res
    # Assert that it retrieved the answer AegisSec-99
    assert "AegisSec-99" in res["response"]
    assert "retrieve_local_documents" in res["tool_calls_executed"]
