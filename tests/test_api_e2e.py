import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_e2e_query_no_search():
    """Verify factual query that does not require search is resolved directly."""
    response = client.post(
        "/query",
        json={"message": "What is the capital of France?", "history": []}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "response" in res_data
    assert "Paris" in res_data["response"]

def test_e2e_query_with_search():
    """Verify query that requires search is successfully completed."""
    response = client.post(
        "/query",
        json={"message": "Who won the most recent Super Bowl?", "history": []}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "response" in res_data
    assert len(res_data["response"]) > 0

def test_e2e_query_local_db():
    """Verify query that requires local database context is resolved using retrieve_local_documents."""
    # 1. Ingest document
    ingest_response = client.post(
        "/ingest",
        json={
            "text": "Project Supernova 9 is an internal next-generation quantum-encryption framework code-named Aegis.",
            "metadata": {"source": "e2e_test"}
        }
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] == "success"
    assert ingest_response.json()["chunk_count"] > 0

    # 2. Query document
    response = client.post(
        "/query",
        json={"message": "What is the internal code-name of project Supernova 9?", "history": []}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "response" in res_data
    assert "Aegis" in res_data["response"]
    assert "retrieve_local_documents" in res_data["tool_calls_executed"]

def test_e2e_query_hybrid_bm25_retrieval():
    """Verify that unique keywords not easily found by semantic search are retrieved via BM25 hybrid path."""
    # 1. Ingest document with a very specific unique identifier keyword and a distinct answer code
    ingest_response = client.post(
        "/ingest",
        json={
            "text": "The secret vault access code for project QWERTY998877X is AegisVaultCode.",
            "metadata": {"source": "e2e_hybrid_test"}
        }
    )
    assert ingest_response.status_code == 200
    
    # 2. Query document using the exact unique keyword to fetch the secret vault access code
    response = client.post(
        "/query",
        json={"message": "What is the secret vault access code for project QWERTY998877X?", "history": []}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "response" in res_data
    # Assert the LLM generates the correct answer 'AegisVaultCode' which was not in the query message
    assert "AegisVaultCode" in res_data["response"]
    assert "retrieve_local_documents" in res_data["tool_calls_executed"]
