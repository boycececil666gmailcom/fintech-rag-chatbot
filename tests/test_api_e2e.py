import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_e2e_query_no_search():
    """Verify factual query that does not require search is resolved directly (and politely refused if unrelated)."""
    response = client.post(
        "/query",
        json={"message": "What is the capital of France?", "history": []}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "response" in res_data
    assert "Fintech" in res_data["response"]



def test_e2e_query_local_db():
    """Verify query that requires local database context is resolved using retrieve_local_documents."""
    # 1. Ingest document
    ingest_response = client.post(
        "/ingest",
        json={
            "text": "Our Fintech SaaS platform allows instant wire transfers up to a daily limit of $10,000.",
            "metadata": {"source": "e2e_test"}
        }
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] == "success"
    assert ingest_response.json()["chunk_count"] > 0

    # 2. Query document
    response = client.post(
        "/query",
        json={"message": "What is the daily wire transfer limit on the SaaS platform?", "history": []}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "response" in res_data
    assert "10,000" in res_data["response"]
    assert "retrieve_local_documents" in res_data["tool_calls_executed"]

def test_e2e_query_hybrid_bm25_retrieval():
    """Verify that unique keywords not easily found by semantic search are retrieved via BM25 hybrid path."""
    # 1. Ingest document with a very specific unique identifier keyword and a distinct answer code
    ingest_response = client.post(
        "/ingest",
        json={
            "text": "Our Fintech SaaS platform uses a proprietary transaction routing mechanism code-named AegisSec-99.",
            "metadata": {"source": "e2e_hybrid_test"}
        }
    )
    assert ingest_response.status_code == 200
    
    # 2. Query document using the exact unique keyword to fetch the secret routing verification code
    response = client.post(
        "/query",
        json={"message": "What is the routing mechanism code name used by the SaaS platform?", "history": []}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "response" in res_data
    # Assert the LLM generates the correct answer 'AegisSec-99' which was not in the query message
    assert "AegisSec-99" in res_data["response"]
    assert "retrieve_local_documents" in res_data["tool_calls_executed"]
